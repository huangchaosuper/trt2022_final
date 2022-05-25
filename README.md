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

---

以下内容待更新

---

## 原始模型
### 模型简介
请介绍模型的基本信息，可以包含但不限于以下内容：
- 用途以及效果
- 业界实际运用情况，比如哪些厂商、哪些产品在用
- 模型的整体结构，尤其是有特色的部分

### 模型优化的难点
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
