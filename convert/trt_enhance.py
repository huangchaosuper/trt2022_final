import os

import tensorrt as trt


class TrtEnhance(object):
    def __init__(self, onnx_model_path, trt_model_path):
        self.onnx_model_file = os.path.join(onnx_model_path, "TrOCR.onnx")
        self.trt_model_path = trt_model_path
        self.logger = trt.Logger(trt.Logger.ERROR)
        pass

    def convert(self, rebuild=True,):
        self.build_tf32_network(rebuild)
        self.build_fp16_mix_network(rebuild)

    def get_plugin_creator(self, plugin_name):
        trt.init_libnvinfer_plugins(self.logger, '')
        plugin_creator_list = trt.get_plugin_registry().plugin_creator_list
        creator = None
        for c in plugin_creator_list:
            if c.name == plugin_name:
                creator = c
        return creator

    def build_tf32_network(self, rebuild):
        trt_model_file = os.path.join(self.trt_model_path, "TrOCR.tf32.plan")
        if rebuild:
            builder = trt.Builder(self.logger)
            plugin_creator = self.get_plugin_creator('LayerNorm')
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            profile = builder.create_optimization_profile()
            if plugin_creator is None:
                print("Plugin CustomPlugin not found.")
                exit()
            parser = trt.OnnxParser(network, self.logger)
            if not os.path.exists(self.onnx_model_file):
                print("Failed finding onnx file!")
                exit()
            print("Succeeded finding onnx file!")
            with open(self.onnx_model_file, 'rb') as model:
                if not parser.parse(model.read()):
                    print("Failed parsing .onnx file!")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    exit()
                print("Succeeded parsing .onnx file!")

            config = builder.create_builder_config()
            config.set_tactic_sources(1 << int(trt.TacticSource.CUBLAS) | 1 << int(trt.TacticSource.CUDNN) | 1 << int(
                trt.TacticSource.CUBLAS_LT))
            #config.flags = (1 << int(trt.BuilderFlag.FP16) | 1 << int(trt.BuilderFlag.STRICT_TYPES))
            config.max_workspace_size = 3 << 30
            input_tensor_pixel_values = network.get_input(0)
            input_tensor_decoder_input_ids = network.get_input(1)
            '''
                trtexec \
                 --onnx=./model/onnx/TrOCR.onnx \
                 --saveEngine=./model/trt/TrOCR.tf32.plan \
                 --minShapes=pixel_values:1x3x384x384,decoder_input_ids:1x1 \
                 --optShapes=pixel_values:8x3x384x384,decoder_input_ids:8x1 \
                 --maxShapes=pixel_values:16x3x384x384,decoder_input_ids:16x1 \
                 --workspace=24000
            '''
            profile.set_shape(input_tensor_pixel_values.name, (1, 3, 384, 384), (8, 3, 384, 384), (16, 3, 384, 384))
            profile.set_shape(input_tensor_decoder_input_ids.name, (1, 1), (8, 1), (16, 1))
            config.add_optimization_profile(profile)
            print("Creating Tensorrt Engine")
            engine_string = builder.build_serialized_network(network, config)
            if engine_string is None:
                print("Failed building engine!")
                exit()
            print("Succeeded building engine!")
            with open(trt_model_file, 'wb') as f:
                f.write(engine_string)
            engine = trt.Runtime(self.logger).deserialize_cuda_engine(engine_string)
        else:
            with open(trt_model_file, 'rb') as f:
                engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
            if engine is None:
                print("Failed loading engine!")
                exit()
            print("Succeeded loading engine!")

        return engine

    def build_fp16_mix_network(self, rebuild):

        trt_model_file = os.path.join(self.trt_model_path, "TrOCR.fp16.plan")
        if rebuild:
            builder = trt.Builder(self.logger)
            plugin_creator = self.get_plugin_creator('LayerNorm')
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            profile = builder.create_optimization_profile()
            if plugin_creator is None:
                print("Plugin CustomPlugin not found.")
                exit()
            parser = trt.OnnxParser(network, self.logger)
            if not os.path.exists(self.onnx_model_file):
                print("Failed finding onnx file!")
                exit()
            print("Succeeded finding onnx file!")
            with open(self.onnx_model_file, 'rb') as model:
                if not parser.parse(model.read()):
                    print("Failed parsing .onnx file!")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    exit()
                print("Succeeded parsing .onnx file!")

            config = builder.create_builder_config()
            config.set_tactic_sources(1 << int(trt.TacticSource.CUBLAS) | 1 << int(trt.TacticSource.CUDNN) | 1 << int(
                trt.TacticSource.CUBLAS_LT))
            config.flags = (1 << int(trt.BuilderFlag.FP16) | 1 << int(trt.BuilderFlag.STRICT_TYPES))
            config.max_workspace_size = 3 << 30
            input_tensor_pixel_values = network.get_input(0)
            input_tensor_decoder_input_ids = network.get_input(1)
            '''
                trtexec \
                 --onnx=./model/onnx/TrOCR.onnx \
                 --saveEngine=./model/trt/TrOCR.tf32.plan \
                 --minShapes=pixel_values:1x3x384x384,decoder_input_ids:1x1 \
                 --optShapes=pixel_values:8x3x384x384,decoder_input_ids:8x1 \
                 --maxShapes=pixel_values:16x3x384x384,decoder_input_ids:16x1 \
                 --workspace=24000
            '''
            profile.set_shape(input_tensor_pixel_values.name, (1, 3, 384, 384), (8, 3, 384, 384), (16, 3, 384, 384))
            profile.set_shape(input_tensor_decoder_input_ids.name, (1, 1), (8, 1), (16, 1))
            config.add_optimization_profile(profile)
            print("Creating Tensorrt Engine")
            engine_string = builder.build_serialized_network(network, config)
            if engine_string is None:
                print("Failed building engine!")
                exit()
            print("Succeeded building engine!")
            with open(trt_model_file, 'wb') as f:
                f.write(engine_string)
            engine = trt.Runtime(self.logger).deserialize_cuda_engine(engine_string)
        else:
            with open(trt_model_file, 'rb') as f:
                engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
            if engine is None:
                print("Failed loading engine!")
                exit()
            print("Succeeded loading engine!")

        return engine
