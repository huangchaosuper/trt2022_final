---
## 总述
该项目为TrOCR模型的TensorRT推理版本实现
- 原始模型名称及链接： https://github.com/microsoft/unilm/tree/master/trocr
- 本项目使用模型： https://github.com/chineseocr/trocr-chinese
- 本项目使用测试数据集： https://aistudio.baidu.com/aistudio/datasetdetail/87750
- 优化效果（精度和加速比）： A10环境下：(按照`batchsize = 8`统计 )

  - TF32 精度可接受，性能提升 23.65%
  - FP16 精度可接受，性能提升 51.12%

## 如何编译和运行

### 准备工作 - 安装nvidia-docker

为了在docker中正常使用GPU，请安装nvidia-docker。

- 如果你的系统是Ubuntu Linux
    - 请参考 [Installing Docker and The Docker Utility Engine for NVIDIA GPUs](https://docs.nvidia.com/ai-enterprise/deployment-guide/dg-docker.html) 安装nvidia-docker
- 如果你的系统是Windows 11
    - 请先参考 [Install Ubuntu on WSL2 on Windows 11 with GUI support](https://ubuntu.com/tutorials/install-ubuntu-on-wsl2-on-windows-11-with-gui-support#1-overview) 把WSL设置好
    - 然后参考 [Running Existing GPU Accelerated Containers on WSL 2](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#ch05-running-containers) 安装nvidia-docker

### 启动环境

#### 第一次执行此项目

``clone``本项目后，执行``start.sh``（若为windows，请参考start.sh内容在windows下复现）

#### 第N次执行此项目（N>1）

执行``restart.sh``

#### 清理本地环境

执行``clear.sh``

#### 执行原始模型`helloworld`

确保前面的所有步骤执行成功，并且当前已经在容器内

```commandline
cd /workspace
python3 test.py
```

#### 执行原始模型全量测试数据

确保前面的所有步骤执行成功，并且当前已经在容器内

```commandline
cd /workspace
bash build.sh
```

## 原始模型
### 模型简介

OCR的模型目前业界已经非常成熟，而且识别率已经达到工业要求，甚至超过人眼的识别能力

一般OCR包含两个主要的部分，`文字检测`和`文字识别`，本模型仅涉及识别部分，文字检测可以使用开源的`dbnet`或其他模型串联

识别模型目前业界使用`LSTM`相关模型较多，但是LSTM模型一般会有比较重的后处理，虽然模型推理时间较快，但是后处理例如`CTCDecoder`非常耗时，并且很多时候是在CPU上进行实现

如果按照一篇文档，文字检测到100个目标，则需要对100个目标分别进行文字识别，虽然使用`batchsize`可以提升一定性能，但是整个E2E推理时间仍然较长

随着`transformer`的识别模型识别率达到或超过`LSTM`，`TrOCR`模型也逐渐被业界关注

其主要优势为`transformer`模型的后处理非常轻，在E2E中几乎不占推理时间，缺点也非常明显，模型本身的推理时间比`LSTM`模型要慢很多

对于文字检测目标为100的一篇文档，推理性能提升`10ms`，整个推理过程提升`1s`，在工业界具有重大的商业价值。

本项目的主要目的是使用`nvidia`的`TensorRT`GPU推理引擎，提升`TrOCR`的推理时间，在保证极高准确率的前提下，性能达到工业界的使用水平，使`TrOCR`具有商业使用价值

随着`nvidia`发布最新的`Hopper`框架，对`transformer`模型进一步友好支持，此项目的商业价值也将得到体现。 


## 原始网络结构改造

对原始模型进行调研后，采用以下pytorch导出代码进行模型改造

```python
  torch.onnx.export(
      model,
      onnx_dummy_inputs,
      onnx_file,
      verbose=True,
      opset_version=13,
      do_constant_folding=True,
      input_names=["pixel_values", "decoder_input_ids"],
      output_names=["generated_ids", "last_hidden_state", "pooler_output"],
      dynamic_axes={"pixel_values": {0: "batch_size"}, "decoder_input_ids": {0: "batch_size"},
                    "generated_ids": {0: "batch_size"}, "last_hidden_state": {0: "batch_size"},
                    "pooler_output": {0: "batch_size"}
                    }
      )
```

导出后的模型输入输出结构如下

``` yaml
  inputs:
      pixel_values=bs,3,384,384
      decoder_input_ids=bs,1
  outputs:
      generated_ids=bs,1,11318 #11318 is vocab.json length#
      last_hidden_state=bs,578,384
      pooler_output=bs,384
```


```shell
trtexec \
 --onnx=./model/onnx/TrOCR.onnx \
 --saveEngine=./model/trt/TrOCR.tf32.plan \
 --minShapes=pixel_values:1x3x384x384,decoder_input_ids:1x1 \
 --optShapes=pixel_values:8x3x384x384,decoder_input_ids:8x1 \
 --maxShapes=pixel_values:16x3x384x384,decoder_input_ids:16x1 \
 --workspace=24000

```


## 优化过程

- 进行 pytorch 到 onnx模型 转换，改造模型预处理并适配输入和输出维度
- 采用 TF32和FP16进行模型转换
  - TF32 精度可接受，性能提升 20%，详情见下表
  - FP16 精度溢出，性能提升 60%， 详情见下表

性能提升计算公式：（ORT-X）/ORT × 100% `RXT3050`

| 类型\批次         | 1      | 4      | 8      | 16      |
|---------------|:-------|:-------|:-------|:--------|
| ORT           | 12.591 | 42.517 | 82.653 | 154.984 |
| TRT（TF32）     | 10.332 | 33.778 | 65.690 | 127.713 |
| TRT（TF32）性能提升 | 17.94% | 20.55% | 20.52% | 17.60%  |
| TRT（FP16）     | 5.324  | 17.208 | 32.544 | 63.444  |
| TRT（FP16）性能提升 | 57.72% | 59.52% | 60.62% | 59.06%  |

修正FP16精度溢出问题：

一般情况下FP16精度益处存在于`POW`之类的算子，因为256的平方就65535，达到了溢出的上限， 
通过使用`Netron`工具查看，`POW`算子可以与关联算子组合为`LayerNorm`算子，在融合后的算子内部避免精度溢出。

- 尝试修正FP16精度溢出问题

  - 编写插件`LayerNorm`并在算子内部避免了精度溢出
  - 虽然使用自定义算子导致FP16性能下降17%左右，但是精度提升明显
  - batchsize为8时，性能和精度（FP16）最佳

PS: 实际业务场景时，OCR文字识别服务也是将batchsize设置为8提升服务吞吐量

性能提升计算公式：（ORT-X）/ORT × 100% `A10`

| 类型\批次         | 1      | 4      | 8      | 16      |
|---------------|:-------|:-------|:-------|:--------|
| ORT           | 6.131 | 18.261 | 33.459 | 63.760 |
| TRT（TF32）     | 4.229 | 13.942 | 25.543 | 49.493 |
| TRT（TF32）性能提升 | 31.02% | 23.65% | 23.65% | 22.37%  |
| TRT（FP16）     | 2.532  | 7.599 | 13.749 | 27.372  |
| TRT（FP16）性能提升 | 58.70% | 58.39% | 58.90% | 57.07%  |
| TRT（FP16 修复精度溢出） | 3.723  | 9.332 | 16.354 | 32.654  |
| TRT（FP16 修复精度溢出）性能提升 | 39.27% | 48.89% | 51.12% | 48.79%  |


- 过程文件存储在`model`文件夹
  - onnx目录下TrOCR.onnx为支持FP32，TF32的ONNX模型文件
  - onnx目录下TrOCR4fp16.onnx为支持FP16的ONNX模型文件
  - trt目录下TrOCR.tf32.plan为支持FP32，TF32的TRT模型文件
  - trt目录下TrOCR.fp16.plan为支持FP16的TRT模型文件


## 精度与加速效果

因为pytorch的模型与onnx和trt的模型结构有调整。即在pytorch导出为onnx模型时模型结构有调整，因此仅对onnx和trt版本做比较。

- 在3050显卡上TRT执行效率如下

onnx throughput: 
```python
{1: 12.590796533333334, 4: 42.517999266666656, 8: 82.6528838, 16: 154.98422660000003}
```

TF32 throughput 及 精度对比
```shell
bs: Batch Size
lt: Latency (ms)
tp: throughput (word/s)
a0: maximum of absolute difference of output 0
r0: median of relative difference of output 0
a1: maximum of absolute difference of output 1
r1: median of relative difference of output 1
a2: maximum of absolute difference of output 2
r2: median of relative difference of output 2
----+--------+---------+---------+---------+---------+---------+---------+---------+-------------
  bs|      lt|       tp|       a0|       r0|       a1|       r1|       a2|       r2| output check
----+--------+---------+---------+---------+---------+---------+---------+---------+-------------
        
 1,  10.332,9.678e+01,5.941e-03,1.079e-04,5.159e-02,1.715e-03,4.619e-04,1.356e-03, Good
16, 127.713,1.253e+02,8.695e-03,1.861e-04,2.893e-02,1.425e-03,8.621e-04,1.043e-03, Good
 4,  33.778,1.184e+02,9.278e-03,1.671e-04,3.284e-02,1.486e-03,4.620e-04,9.444e-04, Good
 8,  65.690,1.218e+02,9.148e-03,3.099e-04,3.284e-02,1.317e-03,4.620e-04,7.367e-04, Good

```

FP16 throughput 及 精度对比

```shell
bs: Batch Size
lt: Latency (ms)
tp: throughput (word/s)
a0: maximum of absolute difference of output 0
r0: median of relative difference of output 0
a1: maximum of absolute difference of output 1
r1: median of relative difference of output 1
a2: maximum of absolute difference of output 2
r2: median of relative difference of output 2
----+--------+---------+---------+---------+---------+---------+---------+---------+-------------
  bs|      lt|       tp|       a0|       r0|       a1|       r1|       a2|       r2| output check
----+--------+---------+---------+---------+---------+---------+---------+---------+-------------
        
 1,   5.324,1.878e+02,8.797e+00,2.526e-01,4.281e+00,1.122e+00,5.385e-01,1.247e+00, Bad
16,  63.444,2.522e+02,1.229e+01,3.306e-01,4.957e+00,1.130e+00,5.964e-01,1.243e+00, Bad
 4,  17.208,2.325e+02,1.222e+01,3.209e-01,4.281e+00,1.130e+00,5.948e-01,1.256e+00, Bad
 8,  32.544,2.458e+02,1.229e+01,3.389e-01,4.281e+00,1.129e+00,5.964e-01,1.227e+00, Bad



```



- 在A10显卡上TRT执行效率如下

onnx throughput: 
```python
{1: 6.131140366666667, 4: 18.261202633333333, 8: 33.45934226666667, 16: 63.76021446666666}
```

TF32 throughput 及 精度对比
```shell
  bs: Batch Size
  lt: Latency (ms)
  tp: throughput (word/s)
  a0: maximum of absolute difference of output 0
  r0: median of relative difference of output 0
  a1: maximum of absolute difference of output 1
  r1: median of relative difference of output 1
  a2: maximum of absolute difference of output 2
  r2: median of relative difference of output 2
  ----+--------+---------+---------+---------+---------+---------+---------+---------+-------------
    bs|      lt|       tp|       a0|       r0|       a1|       r1|       a2|       r2| output check
  ----+--------+---------+---------+---------+---------+---------+---------+---------+-------------

   1,   4.229,2.365e+02,5.441e-03,2.674e-04,8.064e-03,1.337e-03,4.502e-04,1.211e-03, Good
  16,  49.493,3.233e+02,8.971e-03,2.058e-04,2.893e-02,1.413e-03,8.622e-04,1.019e-03, Good
   4,  13.942,2.869e+02,6.756e-03,1.022e-04,3.457e-02,1.367e-03,5.184e-04,7.919e-04, Good
   8,  25.543,3.132e+02,1.028e-02,3.502e-04,2.893e-02,1.403e-03,8.550e-04,1.046e-03, Good

```

FP16 throughput 及 精度对比

```shell
  bs: Batch Size
  lt: Latency (ms)
  tp: throughput (word/s)
  a0: maximum of absolute difference of output 0
  r0: median of relative difference of output 0
  a1: maximum of absolute difference of output 1
  r1: median of relative difference of output 1
  a2: maximum of absolute difference of output 2
  r2: median of relative difference of output 2
  ----+--------+---------+---------+---------+---------+---------+---------+---------+-------------
    bs|      lt|       tp|       a0|       r0|       a1|       r1|       a2|       r2| output check
  ----+--------+---------+---------+---------+---------+---------+---------+---------+-------------

   1,   2.532,3.950e+02,7.964e+00,2.103e-01,3.056e+00,1.107e+00,4.955e-01,1.311e+00, Bad
  16,  27.372,5.845e+02,1.228e+01,3.228e-01,4.957e+00,1.130e+00,5.964e-01,1.236e+00, Bad
   4,   7.599,5.264e+02,1.067e+01,2.547e-01,4.670e+00,1.119e+00,5.950e-01,1.227e+00, Bad
   8,  13.749,5.819e+02,1.229e+01,3.165e-01,4.669e+00,1.130e+00,5.965e-01,1.237e+00, Bad

```

FP16 使用自定义LayerNorm插件后，精度差异及 throughput

```shell
  bs: Batch Size
  lt: Latency (ms)
  tp: throughput (word/s)
  a0: maximum of absolute difference of output 0
  r0: median of relative difference of output 0
  a1: maximum of absolute difference of output 1
  r1: median of relative difference of output 1
  a2: maximum of absolute difference of output 2
  r2: median of relative difference of output 2
  ----+--------+---------+---------+---------+---------+---------+---------+---------+-------------
    bs|      lt|       tp|       a0|       r0|       a1|       r1|       a2|       r2| output check
  ----+--------+---------+---------+---------+---------+---------+---------+---------+-------------

   1,   3.723,2.686e+02,7.740e-02,1.561e-03,1.419e-01,1.192e-02,5.586e-03,1.328e-02, Good
  16,  32.654,4.900e+02,8.810e-02,1.323e-03,1.131e+00,1.166e-02,8.839e-03,6.919e-03, Bad
   4,   9.332,4.286e+02,9.352e-02,1.516e-03,3.307e-01,1.147e-02,8.910e-03,9.156e-03, Good
   8,  16.354,4.892e+02,9.131e-02,1.363e-03,3.157e-01,1.150e-02,8.865e-03,7.560e-03, Good
```