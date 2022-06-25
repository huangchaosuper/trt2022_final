wget https://github.com/huangchaosuper/trt2022_final/releases/download/0.0.1.20220525/model.zip
unzip -o model.zip
mkdir -p ./model/onnx
mkdir -p ./model/trt
wget https://github.com/huangchaosuper/trt2022_final/releases/download/0.0.1.20220525/test_data.zip
unzip -o test_data.zip
docker build -t trt2022_final_base:20220625 -f ./dockerfile_base .
docker build -t trt2022_final:20220625 .
docker run --gpus all -it -d --name trt2022_final_20220625 trt2022_final:20220625 bash
docker exec -it trt2022_final_20220625 bash