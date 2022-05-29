import os
import time

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from logger import logger


class Evaluate(object):
    def __init__(self, pretrained_model_path, model_path):
        self.processor = TrOCRProcessor.from_pretrained(pretrained_model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.model.to(self.device)
        pass

    def pytorch_infer(self, images):
        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values
        logger.info("begin")
        generated_ids = self.model.generate(pixel_values[:, :, :].to(self.device))
        logger.info("end")
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        print(generated_text)

    def pytorch_evaluate(self, test_data_path=r'./test_data'):
        images = []
        image_files = []
        batch_size = 16
        times = 0.0
        for filename in os.listdir(test_data_path):
            image_files.append(os.path.join(test_data_path, filename))
        image_file_len = len(image_files)
        logger.info("image files %d" % image_file_len)
        for x in range(0, len(image_files)):
            if len(image_files) > 0 and len(images) < batch_size:
                images.append(Image.open(image_files.pop()).convert("RGB"))
            if len(images) == batch_size:
                start_time = time.time()
                self.pytorch_infer(images)
                stop_time = time.time()
                images.clear()
                times = times + (stop_time - start_time)
            if len(image_files) == 0:
                start_time = time.time()
                self.pytorch_infer(images)
                stop_time = time.time()
                images.clear()
                times = times + (stop_time - start_time)
                break
        print("average time = %f" % (times / image_file_len))