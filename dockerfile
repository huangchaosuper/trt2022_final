FROM trt2022_final_base:20220625
MAINTAINER huangchaosuper@live.cn
RUN mkdir -p /workspace
COPY . /workspace
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt --extra-index-url https://pypi.ngc.nvidia.com
CMD bash
