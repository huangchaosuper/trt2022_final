##若无法识别掉onnxruntime-gpu可能是cpu版本和gpu版本同时安装，请在容器内执行以下重新安装
# pip3 uninstall onnxruntime onnxruntime-gpu
# pip3 install onnxruntime-gpu

rm -f LayerNormPlugin.so
cd ./plugin/LayerNorm && make clean && make && cp LayerNormPlugin.so ../../ && cd ../../

python main.py