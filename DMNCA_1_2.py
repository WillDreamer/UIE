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
from datasets import TrainDataset
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
LEARNING_RATE = args.learning_rate
EPOCHS = args.epochs
GPU = args.gpu
BATCH_SIZE = args.batchsize
IMAGE_SIZE = args.imagesize


def save_deblur_images(images, iteration, epoch,l):
    filename = './checkpoints/' + METHOD + "/epoch" + str(epoch) + "/" + "Iter_" + str(iteration) + str(l)+ "_restore.png"
    torchvision.utils.save_image(images, filename)


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, 0.5 * math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


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


    encoder_lv1_optim = torch.optim.Adam(encoder_lv1.parameters(), lr=LEARNING_RATE)
    encoder_lv1_scheduler = StepLR(encoder_lv1_optim, step_size=1000, gamma=0.1)
    encoder_lv2_optim = torch.optim.Adam(encoder_lv2.parameters(), lr=LEARNING_RATE)
    encoder_lv2_scheduler = StepLR(encoder_lv2_optim, step_size=1000, gamma=0.1)


    decoder_lv1_optim = torch.optim.Adam(decoder_lv1.parameters(), lr=LEARNING_RATE)
    decoder_lv1_scheduler = StepLR(decoder_lv1_optim, step_size=1000, gamma=0.1)
    decoder_lv2_optim = torch.optim.Adam(decoder_lv2.parameters(), lr=LEARNING_RATE)
    decoder_lv2_scheduler = StepLR(decoder_lv2_optim, step_size=1000, gamma=0.1)


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

    if os.path.exists('./checkpoints/' + METHOD) == False:
        os.system('mkdir ./checkpoints/' + METHOD)

    for epoch in range(args.start_epoch, EPOCHS):
        encoder_lv1_scheduler.step(epoch)
        encoder_lv2_scheduler.step(epoch)

        decoder_lv1_scheduler.step(epoch)
        decoder_lv2_scheduler.step(epoch)

        print("Training...")
        train_dataset = TrainDataset(
            blur_image_files='./data/train_raw.txt',
            sharp_image_files='./data/train_reference.txt',
            root_dir='./data/',
            crop=True,
            crop_size=IMAGE_SIZE,
            transform=transforms.Compose([
                transforms.ToTensor()
            ]))
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        start = 0

        for iteration, images in enumerate(train_dataloader):
            mse = nn.MSELoss().cuda(GPU)
            hsv = HSVLoss().cuda(GPU)
            gt = Variable(images['sharp_image'] - 0.5).cuda(GPU)
            H = gt.size(2)
            W = gt.size(3)

            images_lv1 = Variable(images['blur_image'] - 0.5).cuda(GPU)
            images_lv2_1 = images_lv1[:, :, 0:int(H / 2), :]
            images_lv2_2 = images_lv1[:, :, int(H / 2):H, :]

            feature_lv2_1 = encoder_lv2(images_lv2_1)
            feature_lv2_2 = encoder_lv2(images_lv2_2)
            feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2)
            residual_lv2 = decoder_lv2(feature_lv2)

            feature_lv1 = encoder_lv1(images_lv1 + residual_lv2) + feature_lv2
            deblur_image = decoder_lv1(feature_lv1)

            loss_lv1 = mse(255 * deblur_image, 255 * gt)

            loss = loss_lv1
            loss += 100 * hsv(deblur_image).item()
            encoder_lv1.zero_grad()
            encoder_lv2.zero_grad()

            decoder_lv1.zero_grad()
            decoder_lv2.zero_grad()

            loss.backward()
            encoder_lv1_optim.step()
            encoder_lv2_optim.step()

            decoder_lv1_optim.step()
            decoder_lv2_optim.step()

            if (iteration + 1) % 10 == 0:
                stop = time.time()
                print(
                "epoch:", epoch, "iteration:", iteration + 1, "loss:%.4f" % loss.item(), 'time:%.4f' % (stop - start))
                start = time.time()
        if (epoch) % 200 == 0:
            if os.path.exists('./checkpoints/' + METHOD + '/epoch' + str(epoch)) == False:
                os.system('mkdir ./checkpoints/' + METHOD + '/epoch' + str(epoch))

            print("Testing...")
            test_dataset = TrainDataset(
                blur_image_files='./data/test_raw.txt',
                sharp_image_files='./data/test_reference.txt',
                root_dir='./data/',
                resize=True,
                d=32,
                transform=transforms.Compose([
                    transforms.ToTensor()
                ]))
            test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            test_time = 0.0
            te_loss = 0
            psnr_test = 0
            ssim_test = 0
            for iteration, images in enumerate(test_dataloader):
                with torch.no_grad():
                    test = Variable(images['sharp_image'] - 0.5).cuda(GPU)
                    images_lv1 = Variable(images['blur_image'] - 0.5).cuda(GPU)
                    start = time.time()
                    H = images_lv1.size(2)
                    W = images_lv1.size(3)
                    images_lv2_1 = images_lv1[:, :, 0:int(H / 2), :]
                    images_lv2_2 = images_lv1[:, :, int(H / 2):H, :]

                    feature_lv2_1 = encoder_lv2(images_lv2_1)
                    feature_lv2_2 = encoder_lv2(images_lv2_2)
                    feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2)
                    residual_lv2 = decoder_lv2(feature_lv2)
                    l2 = residual_lv2

                    feature_lv1 = encoder_lv1(images_lv1 + residual_lv2) + feature_lv2
                    deblur_image = decoder_lv1(feature_lv1)
                    te_loss += mse(255 * deblur_image, 255 * test).item()
                    psnr_test += batch_PSNR(deblur_image, test, 1.)
                    ssim_test += batch_SSIM(deblur_image, test, 1.)
                    stop = time.time()
                    test_time += stop - start

                    save_deblur_images(deblur_image.data + 0.5, iteration, epoch,1)
            print('MSE loss:%.4f' % (te_loss / len(test_dataloader)), 'PSNR:%.4f' % (psnr_test / len(test_dataloader)),
                  'SSIM:%.4f' % (ssim_test / len(test_dataloader)))

        torch.save(encoder_lv1.state_dict(), str('./checkpoints/' + METHOD + "/encoder_lv1.pkl"))
        torch.save(encoder_lv2.state_dict(), str('./checkpoints/' + METHOD + "/encoder_lv2.pkl"))

        torch.save(decoder_lv1.state_dict(), str('./checkpoints/' + METHOD + "/decoder_lv1.pkl"))
        torch.save(decoder_lv2.state_dict(), str('./checkpoints/' + METHOD + "/decoder_lv2.pkl"))


if __name__ == '__main__':
    main()





