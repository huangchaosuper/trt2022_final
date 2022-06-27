import time
from PIL import Image

from convert.onnx_enhance import OnnxEnhance

if __name__ == '__main__':
    convert_onnx = OnnxEnhance(r"./model/weights_base", r"./model/weights_base", r"./model/onnx")
    path = r'./assert'
    start_time = time.time()
    convert_onnx.infer(Image.open("./assert/test.png"))
    stop_time = time.time()

    print("average time = %f" % (stop_time - start_time))
    pass
