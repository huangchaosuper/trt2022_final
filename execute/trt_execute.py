import os
import time

import numpy as np

import tensorrt as trt
from cuda import cudart
from glob import glob


class TrtExecute(object):
    def __init__(self, trt_model_path, trt_model_file):
        self.trt_model_file = os.path.join(trt_model_path, trt_model_file)
        logger = trt.Logger(trt.Logger.ERROR)
        with open(self.trt_model_file, 'rb') as engine_file:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(engine_file.read())
        if self.engine is None:
            print("Failed loading %s" % self.trt_model_file)
        print("Succeeded loading %s" % self.trt_model_file)
        self.table_head = \
            """
        bs: Batch Size
        lt: Latency (ms)
        tp: throughput (word/s)
        a0: maximum of absolute difference of output 0
        r0: median of relative difference of output 0
        a1: maximum of absolute difference of output 1
        r1: median of relative difference of output 1
        a2: maximum of absolute difference of output 1
        r2: median of relative difference of output 1
        ----+--------+---------+---------+---------+---------+---------+---------+---------+-------------
          bs|      lt|       tp|       a0|       r0|       a1|       r1|       a2|       r2| output check
        ----+--------+---------+---------+---------+---------+---------+---------+---------+-------------
        """
        pass

    def evaluation(self, save_data_path):
        self.execute(save_data_path)
        pass

    def execute(self, save_data_path):

        nInput = np.sum([self.engine.binding_is_input(i) for i in range(self.engine.num_bindings)])
        nOutput = self.engine.num_bindings - nInput
        context = self.engine.create_execution_context()

        print(self.table_head)  # for standard output

        for ioFile in sorted(glob(save_data_path + "./data-*.npz")):
            io_data = np.load(ioFile, allow_pickle=True)
            pixel_values = io_data['pixel_values']
            decoder_input_ids = io_data['decoder_input_ids']
            batch_size, _ = decoder_input_ids.shape

            context.set_binding_shape(0, pixel_values.shape)
            context.set_binding_shape(1, decoder_input_ids.shape)
            #for i in range(nInput + nOutput):
            #    print("Input ->" if self.engine.binding_is_input(i) else "Output->", self.engine.get_binding_dtype(i),
            #          self.engine.get_binding_shape(i), context.get_binding_shape(i), self.engine.get_binding_dtype(i),
            #          self.engine.get_binding_name(i))
            #print("Finish all input binding: %s" % context.all_binding_shapes_specified)

            bufferH = []
            bufferH.append(pixel_values.astype(np.float32).reshape(-1))
            bufferH.append(decoder_input_ids.astype(np.int64).reshape(-1))
            for i in range(nInput, nInput + nOutput):
                bufferH.append(
                    np.empty(context.get_binding_shape(i), dtype=trt.nptype(self.engine.get_binding_dtype(i))))

            bufferD = []
            for i in range(nInput + nOutput):
                bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

            for i in range(nInput):
                cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes,
                                  cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

            context.execute_v2(bufferD)

            for i in range(nInput, nInput + nOutput):
                cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes,
                                  cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

            # warm up
            for i in range(10):
                context.execute_v2(bufferD)

            # test infernece time
            t0 = time.time_ns()
            for i in range(30):
                context.execute_v2(bufferD)
            t1 = time.time_ns()
            timePerInference = (t1 - t0) / 1000 / 1000 / 30

            generated_ids = self.engine.get_binding_index('generated_ids')
            last_hidden_state = self.engine.get_binding_index('last_hidden_state')
            pooler_output = self.engine.get_binding_index('pooler_output')

            check0 = self.check(bufferH[generated_ids], io_data['generated_ids'], True, 5e-5)
            check1 = self.check(bufferH[last_hidden_state], io_data['last_hidden_state'], True, 5e-5)
            check2 = self.check(bufferH[pooler_output], io_data['pooler_output'], True, 5e-5)

            string = "%4d,%8.3f,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e, %s" % (batch_size,
                                                                                  timePerInference,
                                                                                  batch_size / timePerInference * 1000,
                                                                                  check0[1], check0[2], check1[1],
                                                                                  check1[2], check2[1], check2[2],
                                                                                  "Good" if check0[1] < 3.5e-2 and
                                                                                            check0[2] < 2e-3 and check1[
                                                                                                1] < 1e-1 and check1[
                                                                                                2] < 1e-1 and check2[
                                                                                                1] < 1e-1 and check2[
                                                                                                2] < 1e-1 else "Bad")
            print(string)

            for i in range(nInput + nOutput):
                cudart.cudaFree(bufferD[i])

    def check(self, a, b, weak=False, epsilon=1e-5):
        if weak:
            res = np.all(np.abs(a - b) < epsilon)
        else:
            res = np.all(a == b)
        diff0 = np.max(np.abs(a - b))
        diff1 = np.median(np.abs(a - b) / (np.abs(b) + epsilon))
        # print("check:",res,diff0,diff1)
        return res, diff0, diff1
