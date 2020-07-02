from PIL import Image
import torchvision.transforms as T

##### you need import resnet50
from resnet50b import *

from torch.autograd import Variable as V
import torch as t
from torchvision import transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
trans = T.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    normalize,
])

##### you need write img path here
img = Image.open('/home/user/Work/mjw/tensorrt/agx_tensorrt_merge/tensorrt_car_rec_resnet50/car_rec_test/1*12560*17564.jpg')

make_num=43

input = trans(img)
input = input.unsqueeze(0)
model = resnet50(num_fc0=6,num_fc1=make_num,num_fc2=8).cuda()
model.eval()

##### you need write model path here
model.load_state_dict(t.load('./diyierpi_model_best.pth.tar')['state_dict'])

input = V(input.cuda())
score = model(input)
probability = t.nn.functional.softmax(score[0], dim=1)
max_value, index = t.max(probability, 1)
print("type: ")
print(index,max_value)
probability = t.nn.functional.softmax(score[1], dim=1)
max_value, index = t.max(probability, 1)
print("make: ")
print(index,max_value)
probability = t.nn.functional.softmax(score[2], dim=1)
max_value, index = t.max(probability, 1)
print("color: ")
print(index,max_value)
