import os

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, BertTokenizer

from PIL import Image
from logger import logger


class ConvertOnnx(object):
    def __init__(self, pretrained_model_path, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pretrained_model_path = pretrained_model_path
        self.model_path = model_path
        self.batch_size = 10
        self.decoder_start_token_id = 0
        ##self.processor = TrOCRProcessor.from_pretrained(pretrained_model_path)
        # self.processor = TrOCRProcessor.from_pretrained('microsoft/trocar-small-printed',cache_dir="./working_dir")
        # self.vocab = self.processor.tokenizer.get_vocab()
        ##self.model = VisionEncoderDecoderModel.from_pretrained(model_path, torchscript=True)
        # self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-printed',cache_dir="./working_dir")

        pass

    def convert(self, test_data_path=r"./assert"):
        images = []
        for filename in os.listdir(test_data_path):
            images.append(Image.open(os.path.join(test_data_path, filename)).convert("RGB"))
            if len(images) >= self.batch_size:
                break
        self.onnx_export(images)

    def infer(self, images):
        processor = TrOCRProcessor.from_pretrained(self.pretrained_model_path)
        model = VisionEncoderDecoderModel.from_pretrained(self.model_path, torchscript=True)
        model.eval()
        model.to(self.device)
        pixel_values = processor(images=images, return_tensors="pt").pixel_values
        logger.info("begin")
        generated_ids = model.generate(pixel_values[:, :, :].to(self.device))
        logger.info("end")
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        print(generated_text)

    def onnx_export(self, images):
        processor = TrOCRProcessor.from_pretrained(self.pretrained_model_path)
        model = VisionEncoderDecoderModel.from_pretrained(self.model_path, torchscript=True)
        model.eval()
        model.to("cpu")
        pixel_values = processor(images=images, return_tensors="pt").pixel_values

        decoder_input_ids = torch.ones((self.batch_size, 1), dtype=torch.long,
                                       device="cpu") * self.decoder_start_token_id

        onnx_file = "./model/onnx/TrOCR.onnx"
        onnx_dummy_inputs = (
            pixel_values,
            decoder_input_ids,
        )
        torch.onnx.export(
            model,
            onnx_dummy_inputs,
            onnx_file,
            verbose=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=["pixel_values", "decoder_input_ids"],
            output_names=["generated_ids", "last_hidden_state", "pooler_output"],
            dynamic_axes={"pixel_values": {0: "batch_size"}, "decoder_input_ids": {0: "batch_size"},
                          "generated_ids": {0: "batch_size"}, "last_hidden_state": {0: "batch_size"},
                          "pooler_output": {0: "batch_size"}
                          }
        )
        '''
            inputs:
                pixel_values=bs,3,384,384
                decoder_input_ids=bs,1
            outputs:
                generated_ids=bs,1,11318 #11318 is vocab.json length#
                last_hidden_state=bs,578,384
                pooler_output=bs,384
        '''
        import onnx
        net = onnx.load(onnx_file)
        onnx.checker.check_model(net)
