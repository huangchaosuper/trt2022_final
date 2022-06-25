FROM trt2022_final_base:20220625
MAINTAINER huangchaosuper@live.cn
RUN mkdir -p /workspace
COPY . /workspace
CMD bash
