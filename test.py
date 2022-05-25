import time
from enhance.convert_onnx import ConvertOnnx
from PIL import Image

if __name__ == '__main__':
    convert_onnx = ConvertOnnx("model/weights_chineseocr", "./model/weights_chineseocr")
    path = r'./assert'
    start_time = time.time()
    convert_onnx.convert(Image.open("./assert/test.png"))
    stop_time = time.time()

    print("average time = %f" % (stop_time - start_time))
    pass
