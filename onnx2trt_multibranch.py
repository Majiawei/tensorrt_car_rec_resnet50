

# This sample uses an ONNX ResNet50 Model to create a TensorRT Inference Engine
import random
from PIL import Image
import numpy as np
import cv2
import pycuda.driver as cuda
# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.autoinit

import tensorrt as trt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common

engine_file_path="multi_branch_rec.trt"

class ModelData(object):
    MODEL_PATH = "branch_pt_.onnx"
    INPUT_SHAPE = (3, 448, 448)
    BATCH_SIZE = 16
    # We can convert TensorRT data types to numpy types with trt.nptype()
    DTYPE = trt.float32

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# The Onnx path is used for Onnx models.
def build_engine_onnx(model_file):
    def build_engine():
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = common.GiB(1)
            builder.max_batch_size = ModelData.BATCH_SIZE
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
            print('save engine file as '+engine_file_path)
            # return engine
    if os.path.exists(engine_file_path):
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            return engine
    else:
        build_engine()
def load_normalized_test_case(test_image_list):
    # Converts the input image to a CHW Numpy array
    def normalize_image(image):
        # Resize, antialias and transpose the image to CHW.
        c, h, w = ModelData.INPUT_SHAPE
        image_arr = np.asarray(image.resize((w, h), Image.ANTIALIAS)).transpose([2, 0, 1]).astype(trt.nptype(ModelData.DTYPE)).ravel()
        # This particular ResNet50 model requires some preprocessing, specifically, mean normalization.
        return (image_arr / 255.0 - 0.45) / 0.225

    # f_r = open(test_image_txt,'r')
    # lines = f_r.readlines()
    images=[]
    imgname=[]
    for img_path in test_image_list:
        # img_path = line.strip()
        imgname.append(img_path)
        images.append(normalize_image(Image.open(img_path)))

    return images,imgname

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
    # # test_images = data_files[0]
    # # print(test_images)
    # onnx_model_file, labels_file = data_files[1:]
    test_image_list = []
    onnx_model_file = './branch_pt.onnx'
    labels_file = './class_labels.txt'
    for file in os.listdir('./car_rec_test'):
        if file.endswith('.jpg'):
            test_image_list.append('./car_rec_test/'+file)
            pass
        pass
    # print(onnx_model_file)
    labels = open(labels_file, 'r').read().split('\n')

    engine = build_engine_onnx(onnx_model_file)
    # h_input, d_input, h_output0, d_output0, h_output1, d_output1, h_output2, d_output2, stream = allocate_buffers(engine)
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    context = engine.create_execution_context()

    # test_image_txt = 'test.txt'
    images,imgname = load_normalized_test_case(test_image_list)
    inputs[0].host = np.array(images) #
    # test_case = load_normalized_test_case(test_images, h_input)
    # do_inference(context, h_input, d_input, h_output0, d_output0,h_output1, d_output1,h_output2, d_output2, stream)
    trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=ModelData.BATCH_SIZE)

    h_output0 = trt_outputs[0].reshape([-1, 6])
    h_output1 = trt_outputs[1].reshape([-1, 463])
    h_output2 = trt_outputs[2].reshape([-1, 8])
    print(h_output0.shape)
    # print(trt_outputs[1].shape)
    # print(trt_outputs[2].shape)
    for i in range(0,ModelData.BATCH_SIZE):

        # print(len(h_output0))
        # print(len(h_output1))
        # print(len(h_output2))
        print("----------------")

        # print(imgname[i])

        prob0 = np.max(softmax(h_output0[i]))
        pred0 = new_td[np.argmax(softmax(h_output0[i]))]
        print("type: ", pred0,prob0)

        prob1 = np.max(softmax(h_output1[i]))
        pred1 = labels[np.argmax(softmax(h_output1[i]))]
        print("makemodel: ", pred1,prob1)

        prob2 = np.max(softmax(h_output2[i]))
        pred2 = new_cd[np.argmax(softmax(h_output2[i]))]
        print("color: ", pred2,prob2)
        print("----------------\n")
    


if __name__ == '__main__':
    main()