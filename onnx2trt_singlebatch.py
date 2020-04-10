#
# Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.
#

# This sample uses an ONNX ResNet50 Model to create a TensorRT Inference Engine
import random
from PIL import Image
import numpy as np

import pycuda.driver as cuda
# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.autoinit

import tensorrt as trt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common

engine_file_path="branch_pt_.trt"

class ModelData(object):
    MODEL_PATH = "branch_pt_.onnx"
    INPUT_SHAPE = (3, 448, 448)
    # We can convert TensorRT data types to numpy types with trt.nptype()
    DTYPE = trt.float32

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Allocate host and device buffers, and create a stream.
def allocate_buffers(engine):
    # Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host inputs/outputs.
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(ModelData.DTYPE))
    h_output0 = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(ModelData.DTYPE))
    h_output1 = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(2)), dtype=trt.nptype(ModelData.DTYPE))
    h_output2 = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(3)), dtype=trt.nptype(ModelData.DTYPE))
    # Allocate device memory for inputs and outputs.
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output0 = cuda.mem_alloc(h_output0.nbytes)
    d_output1 = cuda.mem_alloc(h_output1.nbytes)
    d_output2 = cuda.mem_alloc(h_output2.nbytes)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    return h_input, d_input, h_output0, d_output0, h_output1, d_output1, h_output2, d_output2, stream

def do_inference(context, h_input, d_input, h_output0, d_output0, h_output1, d_output1, h_output2, d_output2, stream):
    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(d_input, h_input, stream)
    # Run inference.
    context.execute_async(bindings=[int(d_input), int(d_output0), int(d_output1), int(d_output2)], stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(h_output0, d_output0, stream)
    cuda.memcpy_dtoh_async(h_output1, d_output1, stream)
    cuda.memcpy_dtoh_async(h_output2, d_output2, stream)
    # Synchronize the stream
    stream.synchronize()
    

# The Onnx path is used for Onnx models.
def build_engine_onnx(model_file):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = common.GiB(1)
        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(model_file, 'rb') as model:
            parser.parse(model.read())
        #     print(parser.parse(model.read()))
        #     print(parser.num_errors)
        #     print(parser.get_error(0))
        # last_layer0 = network.get_layer(network.num_layers -11) #output 6
        # last_layer1 = network.get_layer(network.num_layers -6) #185 463
        # last_layer2 = network.get_layer(network.num_layers -1) #187 8

        # print("-------")
        # print(last_layer.get_output(0).name)
        # print(last_layer.get_output(0).shape)
        # print("-------")

        # network.mark_output(last_layer0.get_output(0))
        # network.mark_output(last_layer1.get_output(0))
        # network.mark_output(last_layer2.get_output(0))

        engine = builder.build_cuda_engine(network)
        f = open(engine_file_path, "wb")
        f.write(engine.serialize())
        return engine

def load_normalized_test_case(test_image, pagelocked_buffer):
    # Converts the input image to a CHW Numpy array
    def normalize_image(image):
        # Resize, antialias and transpose the image to CHW.
        c, h, w = ModelData.INPUT_SHAPE
        image_arr = np.asarray(image.resize((w, h), Image.ANTIALIAS)).transpose([2, 0, 1]).astype(trt.nptype(ModelData.DTYPE)).ravel()
        # This particular ResNet50 model requires some preprocessing, specifically, mean normalization.
        return (image_arr / 255.0 - 0.45) / 0.225

    # Normalize the image and copy to pagelocked memory.
    np.copyto(pagelocked_buffer, normalize_image(Image.open(test_image)))
    return test_image

def softmax(x):

    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

cd={"黑":0,"蓝":1,"青":2,"灰":3,"绿":4,"红":5,"白":6,"黄":7 }
td={"sedan":0,"suv":1,"mini_bus":2,"bus":3,"truck":4,"van":5}

new_cd = {v : k for k, v in cd.items()}
new_td = {v : k for k, v in td.items()}

def main():
    # # Set the data path to the directory that contains the trained models and test images for inference.
    # data_path, data_files = common.find_sample_data(description="Runs a ResNet50 network with a TensorRT inference engine.", subfolder="", find_files=["test.jpg", ModelData.MODEL_PATH, "class_labels.txt"])
    # # Get test images, models and labels.
    # # print(len(data_files))
    # # test_images = data_files[0:3]
    # # onnx_model_file, labels_file = data_files[3:]
    # test_images = data_files[0]
    # # print(test_images)
    # onnx_model_file, labels_file = data_files[1:]
    test_images = '/home/user/Desktop/trt2mjw/11131_9.jpg'
    onnx_model_file = '/home/user/Desktop/trt2mjw/new/branch_pt.onnx'
    labels_file = '/home/user/Desktop/trt2mjw/new/class_labels.txt'
    # print(onnx_model_file)
    labels = open(labels_file, 'r').read().split('\n')

    engine = build_engine_onnx(onnx_model_file)
    h_input, d_input, h_output0, d_output0, h_output1, d_output1, h_output2, d_output2, stream = allocate_buffers(engine)
    context = engine.create_execution_context()

    test_case = load_normalized_test_case(test_images, h_input)
    do_inference(context, h_input, d_input, h_output0, d_output0,h_output1, d_output1,h_output2, d_output2, stream)
    # print(len(h_output0))
    # print(len(h_output1))
    # print(len(h_output2))

    prob0 = np.max(softmax(h_output0))
    pred0 = new_td[np.argmax(softmax(h_output0))]
    print("type: ", pred0,prob0)

    prob1 = np.max(softmax(h_output1))
    pred1 = labels[np.argmax(softmax(h_output1))]
    print("makemodel: ", pred1,prob1)

    prob2 = np.max(softmax(h_output2))
    pred2 = new_cd[np.argmax(softmax(h_output2))]
    print("color: ", pred2,prob2)
    
    # if "_".join(pred0.split()) in os.path.splitext(os.path.basename(test_case))[0]:
    #     print("Correctly recognized " + test_case + " as " + pred0)
    # else:
    #     print("Incorrectly recognized " + test_case + " as " + pred0)

if __name__ == '__main__':
    main()