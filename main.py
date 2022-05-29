from enhance.convert_onnx import ConvertOnnx
from enhance.evaluate import Evaluate

if __name__ == '__main__':
    #evaluate = Evaluate(r"./model/weights_chineseocr", r"./model/weights_chineseocr")
    #evaluate.pytorch_evaluate(r"./assert")
    convert_onnx = ConvertOnnx(r"./model/weights_chineseocr", r"./model/weights_chineseocr")
    convert_onnx.convert(r'./assert')
    pass
