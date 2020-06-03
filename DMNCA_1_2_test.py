import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import math
import argparse
import random
from utils import *
import models_ca as models
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from datasets import GoProDataset
import time

parser = argparse.ArgumentParser(description="Deep Multi-Patch Hierarchical Network")
parser.add_argument("-e", "--epochs", type=int, default=2401)
parser.add_argument("-se", "--start_epoch", type=int, default=0)
parser.add_argument("-b", "--batchsize", type=int, default=16)
parser.add_argument("-s", "--imagesize", type=int, default=112)
parser.add_argument("-l", "--learning_rate", type=float, default=0.0001)
parser.add_argument("-g", "--gpu", type=int, default=0)
args = parser.parse_args()

# Hyper Parameters
METHOD = "DMNCA_1_2"
SAMPLE_DIR = "UIQS/D"
EXPDIR = "DMNCA_1_2_test_res"
GPU = args.gpu
def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, 0.5*math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def save_deblur_images(images, iteration, epoch,l):
    filename = './checkpoints/' + METHOD + "/epoch" + str(epoch) + "/" + "Iter_" + str(iteration) + str(l)+ "_restore.png"
    torchvision.utils.save_image(images, filename)


def main():
    print("init data folders")

    encoder_lv1 = models.Encoder()
    encoder_lv2 = models.Encoder()

    decoder_lv1 = models.Decoder()
    decoder_lv2 = models.Decoder()

    encoder_lv1.apply(weight_init).cuda(GPU)
    encoder_lv2.apply(weight_init).cuda(GPU)


    decoder_lv1.apply(weight_init).cuda(GPU)
    decoder_lv2.apply(weight_init).cuda(GPU)

    if os.path.exists(str('./checkpoints/' + METHOD + "/encoder_lv1.pkl")):
        encoder_lv1.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/encoder_lv1.pkl")))
        print("load encoder_lv1 success")
    if os.path.exists(str('./checkpoints/' + METHOD + "/encoder_lv2.pkl")):
        encoder_lv2.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/encoder_lv2.pkl")))
        print("load encoder_lv2 success")

    if os.path.exists(str('./checkpoints/' + METHOD + "/decoder_lv1.pkl")):
        decoder_lv1.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/decoder_lv1.pkl")))
        print("load encoder_lv1 success")
    if os.path.exists(str('./checkpoints/' + METHOD + "/decoder_lv2.pkl")):
        decoder_lv2.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/decoder_lv2.pkl")))
        print("load decoder_lv2 success")


    if os.path.exists('./test_results/' + EXPDIR) == False:
        os.system('mkdir ./test_results/' + EXPDIR)

    iteration = 0.0
    test_time = 0.0
    uicm = 0.0
    uism = 0.0
    uiconm = 0.0
    uiqm = 0.0

    for images_name in os.listdir(SAMPLE_DIR):
        with torch.no_grad():
            iteration += 1
            images_lv1 = transforms.ToTensor()(crop_image(Image.open(SAMPLE_DIR + '/' + images_name).convert('RGB')))
            images_lv1 = Variable(images_lv1 - 0.5).unsqueeze(0).cuda(GPU)
            start = time.time()
            H = images_lv1.size(2)
            W = images_lv1.size(3)

            images_lv2_1 = images_lv1[:, :, 0:int(H / 2), :]
            images_lv2_2 = images_lv1[:, :, int(H / 2):H, :]

            feature_lv2_1 = encoder_lv2(images_lv2_1)
            feature_lv2_2 = encoder_lv2(images_lv2_2)
            feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2)
            residual_lv2 = decoder_lv2(feature_lv2)

            feature_lv1 = encoder_lv1(images_lv1 + residual_lv2) + feature_lv2
            deblur_image = decoder_lv1(feature_lv1)

            deblur_image = torch.clamp(deblur_image, -0.5, 0.5)

            stop = time.time()
            test_time += stop - start
            print('[%d/%d]' % (iteration, len(os.listdir(SAMPLE_DIR))), '   RunTime:%.4f' % (stop - start),
                  '  Average Runtime:%.4f' % (test_time / (iteration + 1)))
            # save_images(deblur_image.data + 0.5, images_name)
            out_np = torch_to_np(deblur_image.data + 0.5)
            # save_images(deblur_image.data + 0.5, images_name)
            uicm_, uism_, uiconm_, uiqm_ = getUIQM(255 * out_np)
            uicm += uicm_
            uism += uism_
            uiconm += uiconm_
            uiqm += uiqm_

    print('UICM:%.4f' % (uicm / len(os.listdir(SAMPLE_DIR))), 'UISM:%.4f' % (uism / len(os.listdir(SAMPLE_DIR))),
              'UICONM:%.4f' % (uiconm / len(os.listdir(SAMPLE_DIR))),
              'UIQM:%.4f' % (uiqm / len(os.listdir(SAMPLE_DIR))))


if __name__ == '__main__':
    main()





