import glob
from random import shuffle
import numpy as np
from PIL import Image

import tensorrt as trt

import labels        #from cityscapes evaluation script
import calibrator    #calibrator.py

# MEAN = (71.60167789, 82.09696889, 72.30508881)
MEAN = (0,0,0)
MODEL_DIR = './'
CITYSCAPES_DIR = '/data/Cityscapes/'
TEST_IMAGE = CITYSCAPES_DIR + 'leftImg8bit/val/lindau/lindau_000042_000019_leftImg8bit.png'
CALIBRATION_DATASET_LOC = CITYSCAPES_DIR + 'leftImg8bit/train/*/*.png'

CLASSES = 19
CHANNEL = 3
HEIGHT = 224
WIDTH = 224

def sub_mean_chw(data):
    data = data.transpose((1,2,0)) # CHW -> HWC
    data -= np.array(MEAN) # Broadcast subtract
    data = data.transpose((2,0,1)) # HWC -> CHW
    return data


def normalize_image(image):
	c, h, w = (CHANNEL,HEIGHT,WIDTH)
	return np.asarray(image.resize((w, h), Image.ANTIALIAS)).transpose([2, 0, 1]).astype(trt.nptype(trt.float32)).ravel()
	pass
             
def color_map(output):
    output = output.reshape(CLASSES, HEIGHT, WIDTH)
    out_col = np.zeros(shape=(HEIGHT, WIDTH), dtype=(np.uint8, 3))
    for x in range (WIDTH):
        for y in range (HEIGHT):
            out_col[y,x] = labels.id2label[labels.trainId2label[np.argmax(output[:,y,x])].id].color
    return out_col

def npmax(output):
	print(output)
	return np.argmax(h_output)
	pass
def create_calibration_dataset():
    # Create list of calibration images (filename)
    # This sample code picks 100 images at random from training set
    calibration_files = glob.glob(CALIBRATION_DATASET_LOC)
    shuffle(calibration_files)
    return calibration_files[:100]

def main():
    calibration_files = create_calibration_dataset()

    # Process 5 images at a time for calibration
    # This batch size can be different from MaxBatchSize (1 in this example)
    batchstream = calibrator.ImageBatchStream(5, calibration_files, normalize_image)
    int8_calibrator = calibrator.PythonEntropyCalibrator(["data"], batchstream)

    # Easy to use TensorRT lite package
    engine = trt.lite.Engine(framework="c1",
                            deployfile=MODEL_DIR + "car_rec.prototxt",
                            modelfile=MODEL_DIR + "car_rec.caffemodel",
                            max_batch_size=1,
                            max_workspace_size=(1 << 30),
                            input_nodes={"data":(CHANNEL,HEIGHT,WIDTH)},
                            output_nodes=["prob"],
                            preprocessors={"data":normalize_image},
                            postprocessors={"prob":npmax},
                            data_type=trt.infer.DataType.INT8,
                            calibrator=int8_calibrator,
                            logger_severity=trt.infer.LogSeverity.INFO)
                           
    test_data = calibrator.ImageBatchStream.read_image_chw(TEST_IMAGE)
    out = engine.infer(test_data)[0]
    print(out)
    # test_img = Image.fromarray(out, 'RGB')
    # test_img.show()