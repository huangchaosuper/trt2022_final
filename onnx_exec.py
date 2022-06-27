import ctypes
from convert.onnx_enhance import OnnxEnhance
from execute.onnx_execute import OnnxExecute

if __name__ == '__main__':
    # pt to onnx
    onnx_enhance = OnnxEnhance(r"./model/weights_chineseocr", r"./model/weights_chineseocr", r"./model/onnx")
    onnx_enhance.convert(r'./assert')  # onnx export and verify
    onnx_enhance.enhance()

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
    pass
