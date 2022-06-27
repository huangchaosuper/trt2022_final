import ctypes
from convert.trt_enhance import TrtEnhance
from execute.trt_execute import TrtExecute

ctypes.cdll.LoadLibrary("./LayerNormPlugin.so")
if __name__ == '__main__':
    # onnx to trt.plan
    trt_enhance = TrtEnhance(r"./model/onnx", r'./model/trt')
    trt_enhance.convert(rebuild=True)  # include tf32 and fp16

    # generate trt accuracy
    trt_execute_tf32 = TrtExecute(r'./model/trt', 'TrOCR.tf32.plan')
    trt_execute_tf32.evaluation(r"./data/")
    trt_execute_fp16 = TrtExecute(r'./model/trt', 'TrOCR.fp16.plan')
    trt_execute_fp16.evaluation(r"./data/")
    trt_execute_fp16_plugin = TrtExecute(r'./model/trt', 'TrOCR.fp16.plugin.plan')
    trt_execute_fp16_plugin.evaluation(r"./data/")
    pass
