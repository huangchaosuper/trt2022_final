import ctypes
from convert.onnx_enhance import OnnxEnhance
from convert.trt_enhance import TrtEnhance
from execute.onnx_execute import OnnxExecute
from execute.trt_execute import TrtExecute

ctypes.cdll.LoadLibrary("./LayerNormPlugin.so")
if __name__ == '__main__':
    # pt to onnx
    onnx_enhance = OnnxEnhance(r"./model/weights_chineseocr", r"./model/weights_chineseocr", r"./model/onnx")
    onnx_enhance.convert(r'./assert')  # onnx export and verify
    onnx_enhance.enhance()
    # onnx to trt.plan
    trt_enhance = TrtEnhance(r"./model/onnx", r'./model/trt')
    trt_enhance.convert(rebuild=True)  # include tf32 and fp16

    # generate npz input and output
    onnx_execute = OnnxExecute(r"./model/onnx", r"TrOCR.onnx", r"./assert")
    # generate onnx accuracy baseline
    onnx_execute.generate_data_baseline(r"./data/", batch_size=1)
    onnx_execute.generate_data_baseline(r"./data/", batch_size=4)
    onnx_execute.generate_data_baseline(r"./data/", batch_size=8)
    onnx_execute.generate_data_baseline(r"./data/", batch_size=16)
    # generate onnx time baseline
    onnx_execute.generate_time_baseline(batch_size=1)
    onnx_execute.generate_time_baseline(batch_size=4)
    onnx_execute.generate_time_baseline(batch_size=8)
    onnx_execute.generate_time_baseline(batch_size=16)
    onnx_execute.print_time_baseline()
    # generate trt accuracy
    trt_execute_tf32 = TrtExecute(r'./model/trt', 'TrOCR.tf32.plan')
    trt_execute_tf32.evaluation(r"./data/")
    trt_execute_fp16 = TrtExecute(r'./model/trt', 'TrOCR.fp16.plan')
    trt_execute_fp16.evaluation(r"./data/")
    trt_execute_fp16_plugin = TrtExecute(r'./model/trt', 'TrOCR.fp16.plugin.plan')
    trt_execute_fp16_plugin.evaluation(r"./data/")
    pass
