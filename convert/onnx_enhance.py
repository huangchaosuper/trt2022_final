import os

import torch
from torch import onnx
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx import TensorProto
from onnxsim import simplify
from PIL import Image
from logger import logger


class OnnxEnhance(object):
    def __init__(self, pretrained_model_path, pt_model_path, onnx_model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pretrained_model_path = pretrained_model_path
        self.model_path = pt_model_path
        self.onnx_model_path = onnx_model_path
        self.onnx_file = os.path.join(self.onnx_model_path, "TrOCR_Level1.onnx")
        self.export_model_path = os.path.join(self.onnx_model_path, "TrOCR.onnx")
        self.nLayerNorm = 0
        self.batch_size = 1
        self.decoder_start_token_id = 0
        ##self.processor = TrOCRProcessor.from_pretrained(pretrained_model_path)
        # self.processor = TrOCRProcessor.from_pretrained('microsoft/trocar-small-printed',cache_dir="./working_dir")
        # self.vocab = self.processor.tokenizer.get_vocab()
        ##self.model = VisionEncoderDecoderModel.from_pretrained(model_path, torchscript=True)
        # self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-printed',cache_dir="./working_dir")
        pass

    def enhance(self):
        self.level1_enhance()

    def level1_enhance(self):
        model = onnx.load(self.onnx_file)
        # simplify onnx model --input-shape speech:4,64,80 speech_lengths:4
        graph = gs.import_onnx(model)
        graph = self.graph_replace_layernorm_node(graph)
        graph.fold_constants().toposort().cleanup()  # 常量折叠并进行拓扑排序
        onnx.save(gs.export_onnx(graph), self.export_model_path)

    def graph_replace_layernorm_node(self, graph):
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_38", "Div_47", "Mul_48")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_115", "Div_124", "Mul_125")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_139", "Div_148", "Mul_149")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_216", "Div_225", "Mul_226")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_240", "Div_249", "Mul_250")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_317", "Div_326", "Mul_327")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_341", "Div_350", "Mul_351")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_418", "Div_427", "Mul_428")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_442", "Div_451", "Mul_452")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_519", "Div_528", "Mul_529")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_543", "Div_552", "Mul_553")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_620", "Div_629", "Mul_630")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_644", "Div_653", "Mul_654")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_721", "Div_730", "Mul_731")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_745", "Div_754", "Mul_755")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_822", "Div_831", "Mul_832")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_846", "Div_855", "Mul_856")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_923", "Div_932", "Mul_933")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_947", "Div_956", "Mul_957")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_1024", "Div_1033", "Mul_1034")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_1048", "Div_1057", "Mul_1058")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_1125", "Div_1134", "Mul_1135")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_1149", "Div_1158", "Mul_1159")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_1226", "Div_1235", "Mul_1236")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_1250", "Div_1259", "Mul_1260")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_1281", "Div_1290", "Mul_1291")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_1362", "Div_1371", "Mul_1372")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_1443", "Div_1452", "Mul_1453")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_1460", "Div_1469", "Mul_1470")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_1541", "Div_1550", "Mul_1551")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_1622", "Div_1631", "Mul_1632")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_1622", "Div_1631", "Mul_1632")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_1639", "Div_1648", "Mul_1649")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_1720", "Div_1729", "Mul_1730")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_1801", "Div_1810", "Mul_1811")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_1818", "Div_1827", "Mul_1828")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_1899", "Div_1908", "Mul_1909")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_1980", "Div_1989", "Mul_1990")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_1997", "Div_2006", "Mul_2007")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_2078", "Div_2087", "Mul_2088")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_2159", "Div_2168", "Mul_2169")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_2159", "Div_2168", "Mul_2169")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_2159", "Div_2168", "Mul_2169")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_2176", "Div_2185", "Mul_2186")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_2257", "Div_2266", "Mul_2267")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_2338", "Div_2347", "Mul_2348")
        graph = self.graph_replace_layernorm_node_by_node_name(graph, "Add_2355", "Div_2364", "Mul_2365")
        return graph

    def graph_replace_layernorm_node_by_node_name(self, graph, start_node_name, break_node_name, stop_node_name,
                                                  start_node_type="Add", stop_node_type="Mul"):
        start_node = None
        stop_node = None
        break_node = None
        self.nLayerNorm = self.nLayerNorm + 1
        for node in graph.nodes:
            if node.op == start_node_type and node.name == start_node_name:
                start_node = node
            if node.op == stop_node_type and node.name == stop_node_name:
                stop_node = node
            if node.op == "Div" and node.name == break_node_name:
                break_node = node

        plugin_variable = gs.Variable("LayerNorm-%d" % self.nLayerNorm, np.dtype(np.float32), None)
        plugin_node = gs.Node("LayerNorm", "LayerNorm-%d" % self.nLayerNorm, inputs=[start_node.outputs[0]],
                              outputs=[plugin_variable], attrs={"epsilon": np.array([0.000009999999747378752])})
        graph.nodes.append(plugin_node)
        stop_node.inputs[0] = plugin_variable
        break_node.outputs.clear()
        return graph

    def convert(self, test_data_path=r"./assert"):
        images = []
        for filename in os.listdir(test_data_path):
            images.append(Image.open(os.path.join(test_data_path, filename)).convert("RGB"))
            if len(images) >= self.batch_size:
                break
        #self.infer(images)
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


        onnx_dummy_inputs = (
            pixel_values,
            decoder_input_ids,
        )
        torch.onnx.export(
            model,
            onnx_dummy_inputs,
            self.onnx_file,
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
        net = onnx.load(self.onnx_file)
        onnx.checker.check_model(net)

