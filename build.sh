##若无法识别掉onnxruntime-gpu可能是cpu版本和gpu版本同时安装，请在容器内执行以下重新安装
pip3 uninstall -y -q onnxruntime onnxruntime-gpu
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple onnxruntime-gpu

rm -f LayerNormPlugin.so
cd ./plugin/LayerNorm && make clean && make && cp LayerNormPlugin.so ../../ && cd ../../

python onnx_exec.py
python trt_exec.py