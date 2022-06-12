FROM nvcr.io/nvidia/tensorrt:22.04-py3
MAINTAINER huangchaosuper@live.cn
RUN mkdir -p /workspace
COPY . /workspace
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu115 \
    && pip3 install -r requirements.txt
RUN pip3 install nvidia-tensorrt==8.2.3.0 pycuda==2021.1 cuda-python==11.6.1 onnxruntime-gpu==1.11.1
CMD bash