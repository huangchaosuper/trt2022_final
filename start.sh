wget https://github.com/huangchaosuper/trt2022_final/releases/download/0.0.1.20220525/model.zip
unzip -o model.zip
wget https://github.com/huangchaosuper/trt2022_final/releases/download/0.0.1.20220525/test_data.zip
unzip -o test_data.zip

docker build -t trt2022_final:20220525 .
docker run --gpus all -it -d --name trt2022_final_20220525 trt2022_final:20220525 bash
docker exec -it trt2022_final_20220525 bash