import argparse, os
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.transforms import ToTensor
import torch.nn as nn
from PIL import Image
from os import listdir
from os.path import join
import os
from tqdm import tqdm

parser = argparse.ArgumentParser(description="PyTorch SR")
parser.add_argument("--model", default="./train/ESPCN/17.pth", type=str, help="path to model (default: none)")
parser.add_argument("--testDir", default="../../TEST/urban100/", type=str, help="path to load lr data (default: none)")
parser.add_argument("--resultDir", default="./output/urban100/", type=str, help="path to save sr data (default: none)")
parser.add_argument("--rate", default="2", type=int, help="which rate upsampling")
parser.add_argument("--cuda", action="store_true", help="use cuda?")

all_time = 0
opt = parser.parse_args()
cuda = opt.cuda

print(opt)

if not opt.testDir:
    print("TestDir is musted!")
    SystemExit(1)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.bmp', '.mat'])


def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:, :, [1, 2]] -= 128
    return np.uint8(rgb.dot(xform.T))


images_path = [join(opt.testDir, x) for x in listdir(opt.testDir) if is_image_file(x)]
print("Load images size: ", len(images_path))

model = torch.load(opt.model)['model']

if cuda:
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()
else:
    model = model.cpu()

if not os.path.isdir(opt.resultDir):
    os.mkdir(opt.resultDir)

for image_path in tqdm(images_path):
    im_imput = Image.open(image_path).convert('YCbCr')
    h, w = im_imput.size
    im_imput = im_imput.resize((h // opt.rate, w // opt.rate), Image.BICUBIC)
    filename = image_path.split('/')[-1].split('.')[0]
    y, cb, cr = im_imput.split()
    h, w = y.size

    input = Variable(ToTensor()(y)).view(1, -1, y.size[1], y.size[0]).cuda()
    with torch.no_grad():
        y_sr = model(input)

    ss = y_sr.cpu().data[0].numpy() * 255.0
    ss = ss.clip(0, 255)
    ss = Image.fromarray(np.uint8(ss[0]), mode='L')
    cb_sr = cb.resize(ss.size, Image.BILINEAR)
    cr_sr = cr.resize(ss.size, Image.BILINEAR)

    out_img = Image.merge('YCbCr', [ss, cb_sr, cr_sr]).convert('RGB')
    out_img.save("%s/%s.bmp" % (opt.resultDir, filename))
