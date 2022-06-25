rm -f LayerNormPlugin.so
cd ./plugin/LayerNorm && make clean && make && cp LayerNormPlugin.so ../../ && cd ../../

python main.py