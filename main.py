import os
import time

from logger import logger
from enhance.convert_onnx import ConvertOnnx
from PIL import Image

if __name__ == '__main__':
    convert_onnx = ConvertOnnx("model/weights_chineseocr", "./model/weights_chineseocr")
    images = []
    image_files = []
    batch_size = 16
    path = r'./test_data'
    times = 0.0
    for filename in os.listdir(path):
        image_files.append(os.path.join(path, filename))
    image_file_len = len(image_files)
    logger.info("image files %d" % image_file_len)
    for x in range(0, len(image_files)):
        if len(image_files) > 0 and len(images) < batch_size:
            images.append(Image.open(image_files.pop()).convert("RGB"))
        if len(images) == batch_size:
            start_time = time.time()
            convert_onnx.convert(images)
            stop_time = time.time()
            images.clear()
            times = times + (stop_time - start_time)
        if len(image_files) == 0:
            start_time = time.time()
            convert_onnx.convert(images)
            stop_time = time.time()
            images.clear()
            times = times + (stop_time - start_time)
            break
    print("average time = %f" % times / image_file_len)
    # convert_onnx.onnx_export(image)
    pass
