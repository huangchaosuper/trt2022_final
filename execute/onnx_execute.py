import os
import time

import numpy as np
import torch
import onnxruntime
import torchvision.transforms as transforms
from PIL import Image


class OnnxExecute(object):
    def __init__(self, onnx_model_path, onnx_file, image_path):
        self.onnx_model_file = os.path.join(onnx_model_path, onnx_file)
        self.time_baseline = dict()
        self.images = []
        for filename in os.listdir(image_path):
            self.images.append(Image.open(os.path.join(image_path, filename)).convert("RGB"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.img_transform = transforms.Compose([
            transforms.Resize((384, 384), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.ort_providers = [
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ]

        self.ort_providers_options = [
            {
                'device_id': 0,
                'gpu_mem_limit': 4 * 1024 * 1024 * 1024,
            }, {

            }
        ]

        self.ort_providers = self.ort_providers
        self.ort_providers_options = self.ort_providers_options
        self.session = onnxruntime.InferenceSession(self.onnx_model_file, providers=self.ort_providers)
        self.session.set_providers(self.ort_providers, self.ort_providers_options)

    def generate_data_baseline(self, save_data_path, batch_size=8):
        io_data = self.process4data(batch_size)
        np.savez(os.path.join(save_data_path, "data-" + str(batch_size)),
                 pixel_values=io_data["pixel_values"],
                 decoder_input_ids=io_data["decoder_input_ids"],
                 generated_ids=io_data["generated_ids"],
                 last_hidden_state=io_data["last_hidden_state"],
                 pooler_output=io_data["pooler_output"])

    def process4data(self, batch_size):
        decoder_start_token_id = 0
        decoder_input_ids = torch.ones((batch_size, 1), dtype=torch.long,
                                       device="cpu") * decoder_start_token_id

        image_list = []
        for i in range(batch_size):
            image_list.append(self.img_transform(self.images[i]))
        pixel_values = torch.stack(image_list)
        io_data = dict()
        io_data["pixel_values"] = pixel_values.numpy()
        io_data["decoder_input_ids"] = decoder_input_ids.numpy()
        ort_inputs = {self.session.get_inputs()[0].name: io_data["pixel_values"].astype(np.float32),
                      self.session.get_inputs()[1].name: io_data["decoder_input_ids"].astype(np.int64)}

        outputs = self.session.run(["generated_ids", "last_hidden_state", "pooler_output"], ort_inputs)

        io_data["generated_ids"] = outputs[0]
        io_data["last_hidden_state"] = outputs[1]
        io_data["pooler_output"] = outputs[2]
        return io_data

    def process4time(self, batch_size):
        decoder_start_token_id = 0
        decoder_input_ids = torch.ones((batch_size, 1), dtype=torch.long,
                                       device="cpu") * decoder_start_token_id

        image_list = []
        for i in range(batch_size):
            image_list.append(self.img_transform(self.images[i]))
        pixel_values = torch.stack(image_list)
        io_data = dict()
        io_data["pixel_values"] = pixel_values.numpy()
        io_data["decoder_input_ids"] = decoder_input_ids.numpy()
        ort_inputs = {self.session.get_inputs()[0].name: io_data["pixel_values"].astype(np.float32),
                      self.session.get_inputs()[1].name: io_data["decoder_input_ids"].astype(np.int64)}
        # test infernece time
        for i in range(10):
            outputs = self.session.run(["generated_ids", "last_hidden_state", "pooler_output"], ort_inputs)
        begin_time = time.time_ns()
        for i in range(30):
            outputs = self.session.run(["generated_ids", "last_hidden_state", "pooler_output"], ort_inputs)
        end_time = time.time_ns()
        self.time_baseline[batch_size] = (end_time - begin_time) / 1000 / 1000 / 30
        pass

    def generate_time_baseline(self, batch_size):
        self.process4time(batch_size)
        pass

    def print_time_baseline(self):
        print(self.time_baseline)
        return self.time_baseline
