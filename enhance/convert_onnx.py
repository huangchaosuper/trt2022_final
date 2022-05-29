import os

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from PIL import Image
from logger import logger


class ConvertOnnx(object):
    def __init__(self, pretrained_model_path, model_path):
        self.processor = TrOCRProcessor.from_pretrained(pretrained_model_path)
        # self.processor = TrOCRProcessor.from_pretrained('microsoft/trocar-small-printed',cache_dir="./working_dir")
        # self.vocab = self.processor.tokenizer.get_vocab()
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
        # self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-printed',cache_dir="./working_dir")
        self.model.eval()
        self.model.to("cuda")
        pass

    def convert(self, test_data_path=r"./test_data"):
        images = []
        for filename in os.listdir(test_data_path):
            images.append(Image.open(os.path.join(test_data_path, filename)).convert("RGB"))
        self.infer(images)

    def infer(self, images):
        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values
        logger.info("begin")
        generated_ids = self.model.generate(pixel_values[:, :, :].to("cuda"))
        logger.info("end")
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        print(generated_text)

    def onnx_export(self, image):
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        torch.onnx.export(
            self.model,
            pixel_values,
            "TrOCR.onnx"
        )
