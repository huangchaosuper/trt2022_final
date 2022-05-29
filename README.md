---
## 总述
该项目为TrOCR模型的TensorRT推理版本实现
- 原始模型名称及链接： https://github.com/microsoft/unilm/tree/master/trocr
- 本项目使用模型： https://github.com/chineseocr/trocr-chinese
- 本项目使用测试数据集： https://aistudio.baidu.com/aistudio/datasetdetail/87750
- TBD：优化效果（精度和加速比），average time = 0.07023331837906086

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
python3 main.py
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

随着`nvidia`发布最新的`H100`框架，对`transformer`模型进一步友好支持，此项目的商业价值也将得到体现。 

---

以下内容待更新

---

- 本模型的输入输出数据结构如下

``` yaml
  inputs:
      pixel_values=bs,3,384,384
      decoder_input_ids=bs,1
  outputs:
      generated_ids=bs,1,11318 #11318 is vocab.json length#
      last_hidden_state=bs,578,384
      pooler_output=bs,384
```



### 模型优化的难点
- 模型优化思路为PT->onnx->trt，如何将pt转换为onnx？第一步就需要对原始网络结构进行改造


```shell
trtexec \
 --onnx=./model/onnx/TrOCR.onnx \
 --saveEngine=./model/trt/TrOCR.tf32.plan \
 --minShapes=pixel_values:1x3x384x384,decoder_input_ids:1x1 \
 --optShapes=pixel_values:8x3x384x384,decoder_input_ids:8x1 \
 --maxShapes=pixel_values:16x3x384x384,decoder_input_ids:16x1 \
 --workspace=24000

```

- 在3050显卡上TRT执行效率如下

TF32

```shell
[05/29/2022-21:55:54] [I] === Trace details ===
[05/29/2022-21:55:54] [I] Trace averages of 10 runs:
[05/29/2022-21:55:54] [I] Average on 10 runs - GPU latency: 33.8371 ms - Host latency: 35.1268 ms (end to end 35.1314 ms, enqueue 34.6321 ms)
[05/29/2022-21:55:54] [I] Average on 10 runs - GPU latency: 33.5586 ms - Host latency: 34.8392 ms (end to end 34.8435 ms, enqueue 34.35 ms)
[05/29/2022-21:55:54] [I] Average on 10 runs - GPU latency: 33.717 ms - Host latency: 35.0445 ms (end to end 35.0494 ms, enqueue 34.5316 ms)
[05/29/2022-21:55:54] [I] Average on 10 runs - GPU latency: 33.6805 ms - Host latency: 34.978 ms (end to end 34.9823 ms, enqueue 34.4815 ms)
[05/29/2022-21:55:54] [I] Average on 10 runs - GPU latency: 33.5955 ms - Host latency: 34.8849 ms (end to end 34.8894 ms, enqueue 34.3928 ms)
[05/29/2022-21:55:54] [I] Average on 10 runs - GPU latency: 33.6902 ms - Host latency: 35.0375 ms (end to end 35.0424 ms, enqueue 34.5155 ms)
[05/29/2022-21:55:54] [I] Average on 10 runs - GPU latency: 33.6041 ms - Host latency: 35.0108 ms (end to end 35.016 ms, enqueue 34.4454 ms)
[05/29/2022-21:55:54] [I] Average on 10 runs - GPU latency: 33.6895 ms - Host latency: 35.1783 ms (end to end 35.1835 ms, enqueue 34.6011 ms)
[05/29/2022-21:55:54] [I] 
[05/29/2022-21:55:54] [I] === Performance summary ===
[05/29/2022-21:55:54] [I] Throughput: 28.9757 qps
[05/29/2022-21:55:54] [I] Latency: min = 34.6697 ms, max = 36.1511 ms, mean = 35.0057 ms, median = 34.916 ms, percentile(99%) = 36.1511 ms
[05/29/2022-21:55:54] [I] End-to-End Host Latency: min = 34.6746 ms, max = 36.156 ms, mean = 35.0105 ms, median = 34.9216 ms, percentile(99%) = 36.156 ms
[05/29/2022-21:55:54] [I] Enqueue Time: min = 34.2643 ms, max = 35.6603 ms, mean = 34.4884 ms, median = 34.4119 ms, percentile(99%) = 35.6603 ms
[05/29/2022-21:55:54] [I] H2D Latency: min = 0.788696 ms, max = 1.17773 ms, mean = 0.828755 ms, median = 0.80957 ms, percentile(99%) = 1.17773 ms
[05/29/2022-21:55:54] [I] GPU Compute Time: min = 33.4643 ms, max = 34.8665 ms, mean = 33.664 ms, median = 33.5995 ms, percentile(99%) = 34.8665 ms
[05/29/2022-21:55:54] [I] D2H Latency: min = 0.289551 ms, max = 0.793457 ms, mean = 0.512931 ms, median = 0.497314 ms, percentile(99%) = 0.793457 ms
[05/29/2022-21:55:54] [I] Total Host Walltime: 3.07153 s
[05/29/2022-21:55:54] [I] Total GPU Compute Time: 2.9961 s
[05/29/2022-21:55:54] [W] * Throughput may be bound by Enqueue Time rather than GPU Compute and the GPU may be under-utilized.
[05/29/2022-21:55:54] [W]   If not already in use, --useCudaGraph (utilize CUDA graphs where possible) may increase the throughput.
[05/29/2022-21:55:54] [I] Explanations of the performance metrics are printed in the verbose logs.


[05/29/2022-22:03:19] [I] === Trace details ===
[05/29/2022-22:03:19] [I] Trace averages of 10 runs:
[05/29/2022-22:03:19] [I] Average on 10 runs - GPU latency: 65.2362 ms - Host latency: 67.8229 ms (end to end 67.8285 ms, enqueue 66.8017 ms)
[05/29/2022-22:03:19] [I] Average on 10 runs - GPU latency: 65.1597 ms - Host latency: 67.7441 ms (end to end 67.7487 ms, enqueue 66.7372 ms)
[05/29/2022-22:03:19] [I] Average on 10 runs - GPU latency: 65.2015 ms - Host latency: 67.9667 ms (end to end 67.972 ms, enqueue 66.8608 ms)
[05/29/2022-22:03:19] [I] Average on 10 runs - GPU latency: 65.3969 ms - Host latency: 68.0103 ms (end to end 68.0148 ms, enqueue 66.9929 ms)
[05/29/2022-22:03:19] [I] 
[05/29/2022-22:03:19] [I] === Performance summary ===
[05/29/2022-22:03:19] [I] Throughput: 14.9463 qps
[05/29/2022-22:03:19] [I] Latency: min = 67.447 ms, max = 68.8347 ms, mean = 67.8894 ms, median = 67.7964 ms, percentile(99%) = 68.8347 ms
[05/29/2022-22:03:19] [I] End-to-End Host Latency: min = 67.4512 ms, max = 68.8386 ms, mean = 67.8943 ms, median = 67.8026 ms, percentile(99%) = 68.8386 ms
[05/29/2022-22:03:19] [I] Enqueue Time: min = 66.5347 ms, max = 67.8513 ms, mean = 66.869 ms, median = 66.8098 ms, percentile(99%) = 67.8513 ms
[05/29/2022-22:03:19] [I] H2D Latency: min = 1.53021 ms, max = 1.97266 ms, mean = 1.59597 ms, median = 1.57715 ms, percentile(99%) = 1.97266 ms
[05/29/2022-22:03:19] [I] GPU Compute Time: min = 64.983 ms, max = 66.3162 ms, mean = 65.2789 ms, median = 65.2197 ms, percentile(99%) = 66.3162 ms
[05/29/2022-22:03:19] [I] D2H Latency: min = 0.57666 ms, max = 1.36975 ms, mean = 1.01454 ms, median = 1.00183 ms, percentile(99%) = 1.36975 ms
[05/29/2022-22:03:19] [I] Total Host Walltime: 3.14459 s
[05/29/2022-22:03:19] [I] Total GPU Compute Time: 3.06811 s
[05/29/2022-22:03:19] [W] * Throughput may be bound by Enqueue Time rather than GPU Compute and the GPU may be under-utilized.
[05/29/2022-22:03:19] [W]   If not already in use, --useCudaGraph (utilize CUDA graphs where possible) may increase the throughput.
[05/29/2022-22:03:19] [I] Explanations of the performance metrics are printed in the verbose logs.

```

FP16
```shell
[05/29/2022-22:18:25] [I] === Trace details ===
[05/29/2022-22:18:25] [I] Trace averages of 10 runs:
[05/29/2022-22:18:25] [I] Average on 10 runs - GPU latency: 32.2999 ms - Host latency: 34.8031 ms (end to end 34.8144 ms, enqueue 33.7707 ms)
[05/29/2022-22:18:25] [I] Average on 10 runs - GPU latency: 32.1234 ms - Host latency: 34.618 ms (end to end 34.6294 ms, enqueue 33.5885 ms)
[05/29/2022-22:18:25] [I] Average on 10 runs - GPU latency: 32.4685 ms - Host latency: 34.9879 ms (end to end 35.0003 ms, enqueue 33.9485 ms)
[05/29/2022-22:18:25] [I] Average on 10 runs - GPU latency: 32.3111 ms - Host latency: 34.8197 ms (end to end 34.832 ms, enqueue 33.7871 ms)
[05/29/2022-22:18:25] [I] Average on 10 runs - GPU latency: 32.3106 ms - Host latency: 34.8027 ms (end to end 34.8139 ms, enqueue 33.7733 ms)
[05/29/2022-22:18:25] [I] Average on 10 runs - GPU latency: 32.5087 ms - Host latency: 35.0325 ms (end to end 35.0429 ms, enqueue 33.9902 ms)
[05/29/2022-22:18:25] [I] Average on 10 runs - GPU latency: 32.3241 ms - Host latency: 34.8221 ms (end to end 34.833 ms, enqueue 33.7923 ms)
[05/29/2022-22:18:25] [I] Average on 10 runs - GPU latency: 32.3076 ms - Host latency: 34.7997 ms (end to end 34.8115 ms, enqueue 33.7696 ms)
[05/29/2022-22:18:25] [I] Average on 10 runs - GPU latency: 32.4467 ms - Host latency: 34.9571 ms (end to end 34.9671 ms, enqueue 33.9204 ms)
[05/29/2022-22:18:25] [I] 
[05/29/2022-22:18:25] [I] === Performance summary ===
[05/29/2022-22:18:25] [I] Throughput: 29.5527 qps
[05/29/2022-22:18:25] [I] Latency: min = 34.2395 ms, max = 36.4373 ms, mean = 34.8425 ms, median = 34.7981 ms, percentile(99%) = 36.4373 ms
[05/29/2022-22:18:25] [I] End-to-End Host Latency: min = 34.2493 ms, max = 36.4438 ms, mean = 34.8538 ms, median = 34.8093 ms, percentile(99%) = 36.4438 ms
[05/29/2022-22:18:25] [I] Enqueue Time: min = 33.3905 ms, max = 35.3684 ms, mean = 33.8134 ms, median = 33.7675 ms, percentile(99%) = 35.3684 ms
[05/29/2022-22:18:25] [I] H2D Latency: min = 1.5199 ms, max = 1.57434 ms, mean = 1.53397 ms, median = 1.52936 ms, percentile(99%) = 1.57434 ms
[05/29/2022-22:18:25] [I] GPU Compute Time: min = 31.9314 ms, max = 33.877 ms, mean = 32.3422 ms, median = 32.2915 ms, percentile(99%) = 33.877 ms
[05/29/2022-22:18:25] [I] D2H Latency: min = 0.575684 ms, max = 1.01001 ms, mean = 0.966306 ms, median = 0.967651 ms, percentile(99%) = 1.01001 ms
[05/29/2022-22:18:25] [I] Total Host Walltime: 3.07924 s
[05/29/2022-22:18:25] [I] Total GPU Compute Time: 2.94314 s
[05/29/2022-22:18:25] [W] * Throughput may be bound by Enqueue Time rather than GPU Compute and the GPU may be under-utilized.
[05/29/2022-22:18:25] [W]   If not already in use, --useCudaGraph (utilize CUDA graphs where possible) may increase the throughput.
[05/29/2022-22:18:25] [I] Explanations of the performance metrics are printed in the verbose logs.

```


## 原始网络结构改造

```shell
pip3 install transformers[onnx]
```

```shell
python -m transformers.onnx --model=./model/weights_chineseocr ./model/onnx/
```

如果模型可以容易地跑在TensorRT上而且性能很好，就没有必要选它作为参赛题目并在这里长篇大论了。相信你选择了某个模型作为参赛题目必然有选择它的理由。  
请介绍一下在模型在导出时、或用polygraphy/trtexec解析时、或在TensorRT运行时，会遇到什么问题。换句话说，针对这个模型，我们为什么需要额外的工程手段。

## 优化过程
这一部分是报告的主体。请把自己假定为老师，为TensorRT的初学者讲述如何从原始模型出发，经过一系列开发步骤，得到优化后的TensorRT模型。  

建议：
- 分步骤讲清楚开发过程
- 最好能介绍为什么需要某个特别步骤，通过这个特别步骤解决了什么问题
  - 比如，通过Nsight Systems绘制timeline做了性能分析，发现attention时间占比高且有优化空间（贴图展示分析过程），所以决定要写plugin。然后介绍plugin的设计与实现，并在timeline上显示attention这一部分的性能改进。

## 精度与加速效果
这一部分介绍优化模型在云主机上的运行效果，需要分两部分说明：  
- 精度：报告与原始模型进行精度对比测试的结果，验证精度达标。
  - 这里的精度测试指的是针对“原始模型”和“TensorRT优化模型”分别输出的数据（tensor）进行数值比较。请给出绝对误差和相对误差的统计结果（至少包括最大值、平均值与中位数）。
  - 使用训练好的权重和有意义的输入数据更有说服力。如果选手使用了随机权重和输入数据，请在这里注明。  
  - 在精度损失较大的情况下，鼓励选手用训练好的权重和测试数据集对模型优化前与优化后的准确度指标做全面比较，以增强说服力
- 性能：最好用图表展示不同batch size或sequence length下性能加速效果。
  - 一般用原始模型作为参考标准；若额外使用ONNX Runtime作为参考标准则更好。  
  - 一般提供模型推理时间的加速比即可；若能提供压力测试下的吞吐提升则更好。

请注意：
- 相关测试代码也需要包含在代码仓库中，可被复现。
- 请写明云主机的软件硬件环境，方便他人参考。  

## Bug报告（可选）
