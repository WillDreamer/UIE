import math
import torch
import torch.nn as nn
import numpy as np
from skimage.measure.simple_metrics import compare_psnr
from skimage.measure import compare_ssim
import math
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from PIL import Image
import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F


class HSVLoss(nn.Module):
    def __init__(self, h=0, s=1, v=0.7, eps=1e-7, threshold_h=0.03, threshold_sv=0.1):
        super(HSVLoss, self).__init__()
        self.hsv = [h, s, v]
        self.loss = nn.L1Loss(reduction='none')
        self.eps = eps

        # since Hue is a circle (where value 0 is equal to value 1 that are both "red"),
        # we need a threshold to prevent the gradient explod effect
        # the smaller the threshold, the optimal hue can to more close to target hue
        self.threshold_h = threshold_h
        # since Hue and (Value and Satur) are conflict when generated image' hue is not the target Hue,
        # we have to condition V to prevent it from interfering the Hue loss
        # the larger the threshold, the ealier to activate V loss
        self.threshold_sv = threshold_sv

    def get_hsv(self, im):
        img = im * 0.5 + 0.5
        hue = torch.Tensor(im.shape[0], im.shape[2], im.shape[3]).to(im.device)

        hue[img[:, 2] == img.max(1)[0]] = 4.0 + ((img[:, 0] - img[:, 1]) / (img.max(1)[0] - img.min(1)[0] + self.eps))[
            img[:, 2] == img.max(1)[0]]
        hue[img[:, 1] == img.max(1)[0]] = 2.0 + ((img[:, 2] - img[:, 0]) / (img.max(1)[0] - img.min(1)[0] + self.eps))[
            img[:, 1] == img.max(1)[0]]
        hue[img[:, 0] == img.max(1)[0]] = (0.0 + ((img[:, 1] - img[:, 2]) / (img.max(1)[0] - img.min(1)[0] + self.eps))[
            img[:, 0] == img.max(1)[0]]) % 6

        hue[img.min(1)[0] == img.max(1)[0]] = 0.0
        hue = hue / 6

        saturation = (img.max(1)[0] - img.min(1)[0]) / (img.max(1)[0] + self.eps)
        saturation[img.max(1)[0] == 0] = 0

        value = img.max(1)[0]
        return hue, saturation, value

    def get_rgb_from_hsv(self):
        C = self.hsv[2] * self.hsv[1]
        X = C * (1 - abs((self.hsv[0] * 6) % 2 - 1))
        m = self.hsv[2] - C

        if self.hsv[0] < 1 / 6:
            R_hat, G_hat, B_hat = C, X, 0
        elif self.hsv[0] < 2 / 6:
            R_hat, G_hat, B_hat = X, C, 0
        elif self.hsv[0] < 3 / 6:
            R_hat, G_hat, B_hat = 0, C, X
        elif self.hsv[0] < 4 / 6:
            R_hat, G_hat, B_hat = 0, X, C
        elif self.hsv[0] < 5 / 6:
            R_hat, G_hat, B_hat = X, 0, C
        elif self.hsv[0] <= 6 / 6:
            R_hat, G_hat, B_hat = C, 0, X

        R, G, B = (R_hat + m), (G_hat + m), (B_hat + m)

        return R, G, B

    def forward(self, input):
        h, s, v = self.get_hsv(input)

        target_h = torch.Tensor(h.shape).fill_(self.hsv[0]).to(input.device).type_as(h)
        target_s = torch.Tensor(s.shape).fill_(self.hsv[1]).to(input.device).type_as(s)
        target_v = torch.Tensor(v.shape).fill_(self.hsv[2]).to(input.device).type_as(v)

        loss_h = self.loss(h, target_h)
        loss_h[loss_h < self.threshold_h] = 0.0
        loss_h = loss_h.mean()

        if loss_h < self.threshold_h * 3:
            loss_h = torch.Tensor([0]).to(input.device)

        loss_s = self.loss(s, target_s).mean()
        if loss_h.item() > self.threshold_sv:
            loss_s = torch.Tensor([0]).to(input.device)

        loss_v = self.loss(v, target_v).mean()
        if loss_h.item() > self.threshold_sv:
            loss_v = torch.Tensor([0]).to(input.device)

        return loss_h + 4e-1 * loss_s + 4e-1 * loss_v


def crop_image(img, d=32):
    '''Make dimensions divisible by `d`'''

    new_size = (img.size[0] - img.size[0] % d,
                img.size[1] - img.size[1] % d)

    bbox = [
        int((img.size[0] - new_size[0]) / 2),
        int((img.size[1] - new_size[1]) / 2),
        int((img.size[0] + new_size[0]) / 2),
        int((img.size[1] + new_size[1]) / 2),
    ]

    img_cropped = img.crop(bbox)
    return img_cropped


def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2, 0, 1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.


def np_to_pil(img_np):
    '''Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)


def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.
    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.
    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]


def mu_a(x, alpha_L=0.1, alpha_R=0.1):
    """
      Calculates the asymetric alpha-trimmed mean
    """
    # sort pixels by intensity - for clipping
    x = sorted(x)
    # get number of pixels
    K = len(x)
    # calculate T alpha L and T alpha R
    T_a_L = math.ceil(alpha_L * K)
    T_a_R = math.floor(alpha_R * K)
    # calculate mu_alpha weight
    weight = (1.0 / (K - T_a_L - T_a_R))
    # loop through flattened image starting at T_a_L+1 and ending at K-T_a_R
    s = int(T_a_L + 1)
    e = int(K - T_a_R)
    val = sum(x[s:e])
    val = weight * val
    return val


def s_a(x, mu):
    val = 0
    for pixel in x:
        val += math.pow((pixel - mu), 2)
    return val / len(x)


def _uicm(x):
    R = x[0, :, :].flatten()
    G = x[1, :, :].flatten()
    B = x[2, :, :].flatten()
    RG = R - G
    YB = ((R + G) / 2.0) - B
    mu_a_RG = mu_a(RG)
    mu_a_YB = mu_a(YB)
    s_a_RG = s_a(RG, mu_a_RG)
    s_a_YB = s_a(YB, mu_a_YB)
    l = math.sqrt((math.pow(mu_a_RG, 2) + math.pow(mu_a_YB, 2)))
    r = math.sqrt(s_a_RG + s_a_YB)
    return (-0.0268 * l) + (0.1586 * r)


def sobel(x):
    dx = ndimage.sobel(x, 0)
    dy = ndimage.sobel(x, 1)
    mag = np.hypot(dx, dy)
    mag *= 255.0 / np.max(mag)
    return mag


def eme(x, window_size):
    """
      Enhancement measure estimation
      x.shape[0] = height
      x.shape[1] = width
    """
    # if 4 blocks, then 2x2...etc.
    k1 = int(x.shape[1] / window_size)

    k2 = int(x.shape[0] / window_size)
    # weight
    w = 2. / (k1 * k2)
    blocksize_x = window_size
    blocksize_y = window_size
    # make sure image is divisible by window_size - doesn't matter if we cut out some pixels
    x = x[:blocksize_y * k2, :blocksize_x * k1]
    val = 0
    for l in range(k1):
        for k in range(k2):
            block = x[k * window_size:window_size * (k + 1), l * window_size:window_size * (l + 1)]
            max_ = np.max(block)
            min_ = np.min(block)

            # bound checks, can't do log(0)
            if min_ <= 0.0:
                val += 0
            elif max_ <= 0.0:
                val += 0
            else:
                val += math.log(int(max_ / min_))


    return w * val


def _uism(x):
    """
      Underwater Image Sharpness Measure
    """
    # get image channels
    R = x[0, :, :]
    G = x[1, :, :]
    B = x[2, :, :]
    # first apply Sobel edge detector to each RGB component
    Rs = sobel(R)
    Gs = sobel(G)
    Bs = sobel(B)
    # multiply the edges detected for each channel by the channel itself
    R_edge_map = np.multiply(Rs, R)
    G_edge_map = np.multiply(Gs, G)
    B_edge_map = np.multiply(Bs, B)
    # get eme for each channel
    r_eme = eme(R_edge_map, 10)
    g_eme = eme(G_edge_map, 10)
    b_eme = eme(B_edge_map, 10)
    # coefficients
    lambda_r = 0.299
    lambda_g = 0.587
    lambda_b = 0.114
    return (lambda_r * r_eme) + (lambda_g * g_eme) + (lambda_b * b_eme)


def plip_g(x, mu=1026.0):
    return mu - x


def plip_theta(g1, g2, k):
    g1 = plip_g(g1)
    g2 = plip_g(g2)
    return k * ((g1 - g2) / (k - g2))


def plip_cross(g1, g2, gamma):
    g1 = plip_g(g1)
    g2 = plip_g(g2)
    return g1 + g2 - ((g1 * g2) / (gamma))


def plip_diag(c, g, gamma):
    g = plip_g(g)
    return gamma - (gamma * math.pow((1 - (g / gamma)), c))


def plip_multiplication(g1, g2):
    return plip_phiInverse(plip_phi(g1) * plip_phi(g2))
    # return plip_phiInverse(plip_phi(plip_g(g1)) * plip_phi(plip_g(g2)))


def plip_phiInverse(g):
    plip_lambda = 1026.0
    plip_beta = 1.0
    return plip_lambda * (1 - math.pow(math.exp(-g / plip_lambda), 1 / plip_beta));


def plip_phi(g):
    plip_lambda = 1026.0
    plip_beta = 1.0
    return -plip_lambda * math.pow(math.log(1 - g / plip_lambda), plip_beta)


def _uiconm(x, window_size):
    """
      Underwater image contrast measure
      https://github.com/tkrahn108/UIQM/blob/master/src/uiconm.cpp
      https://ieeexplore.ieee.org/abstract/document/5609219
    """
    plip_lambda = 1026.0
    plip_gamma = 1026.0
    plip_beta = 1.0
    plip_mu = 1026.0
    plip_k = 1026.0
    # if 4 blocks, then 2x2...etc.

    k1 = int(x.shape[1] / window_size)
    k2 = int(x.shape[2] / window_size)
    # weight
    w = -1. / (k1 * k2)
    blocksize_x = window_size
    blocksize_y = window_size

    # entropy scale - higher helps with randomness
    alpha = 1

    R = x[0, :, :]
    G = x[1, :, :]
    B = x[2, :, :]

    # make sure image is divisible by window_size - doesn't matter if we cut out some pixels
    R = R[:blocksize_y * k1, :blocksize_x * k2]

    val_red = 0
    for l in range(k2):
        for k in range(k1):
            block = R[k * window_size:window_size * (k + 1), l * window_size:window_size * (l + 1)]
            max_ = np.max(block)
            min_ = np.min(block)
            top = max_ - min_
            bot = max_ + min_
            if math.isnan(top) or math.isnan(bot) or bot == 0.0 or top == 0.0:
                val_red += 0.0
            else: val_red += alpha*math.pow((top/bot),alpha) * math.log(top/bot)
            #else:
                #val += plip_multiplication((top / bot), math.log(top / bot))
        # make sure image is divisible by window_size - doesn't matter if we cut out some pixels
    G = G[:blocksize_y * k1, :blocksize_x * k2]
    val_green = 0
    for l in range(k2):
        for k in range(k1):
            block = G[k * window_size:window_size * (k + 1), l * window_size:window_size * (l + 1)]
            max_ = np.max(block)
            min_ = np.min(block)
            top = max_ - min_
            bot = max_ + min_
            if math.isnan(top) or math.isnan(bot) or bot == 0.0 or top == 0.0:
                val_green += 0.0
            else:
                val_green += alpha * math.pow((top / bot), alpha) * math.log(top / bot)

    B = B[:blocksize_y * k1, :blocksize_x * k2]
    val_blue = 0
    for l in range(k2):
        for k in range(k1):
            block = B[k * window_size:window_size * (k + 1), l * window_size:window_size * (l + 1)]
            max_ = np.max(block)
            min_ = np.min(block)
            top = max_ - min_
            bot = max_ + min_
            if math.isnan(top) or math.isnan(bot) or bot == 0.0 or top == 0.0:
                val_blue += 0.0
            else:
                val_blue += alpha * math.pow((top / bot), alpha) * math.log(top / bot)
    val = val_blue+val_green+val_red
    return w * val


def getUIQM(x):
    """
      Function to return UIQM to be called from other programs
      x: image
    """
    x = x.astype(np.float32)
    ### UCIQE: https://ieeexplore.ieee.org/abstract/document/7300447
    # c1 = 0.4680; c2 = 0.2745; c3 = 0.2576
    ### UIQM https://ieeexplore.ieee.org/abstract/document/7305804
    c1 = 0.0282;
    c2 = 0.2953;
    c3 = 3.5753
    uicm = _uicm(x)
    uism = _uism(x)
    uiconm = _uiconm(x, 8)
    uiqm = (c1 * uicm) + (c2 * uism) + (c3 * uiconm)
    return uicm, uism, uiconm, uiqm


class Morphology(nn.Module):
    '''
    Base class for morpholigical operators
    For now, only supports stride=1, dilation=1, kernel_size H==W, and padding='same'.
    '''

    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=15, type=None):
        '''
        in_channels: scalar
        out_channels: scalar, the number of the morpholigical neure.
        kernel_size: scalar, the spatial size of the morpholigical neure.
        soft_max: bool, using the soft max rather the torch.max(), ref: Dense Morphological Networks: An Universal Function Approximator (Mondal et al. (2019)).
        beta: scalar, used by soft_max.
        type: str, dilation2d or erosion2d.
        '''
        super(Morphology, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.soft_max = soft_max
        self.beta = beta
        self.type = type

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size),
                                   requires_grad=True)
        self.unfold = nn.Unfold(kernel_size, dilation=1, padding=0, stride=1)

    def forward(self, x):
        '''
        x: tensor of shape (B,C,H,W)
        '''
        # padding
        x = fixed_padding(x, self.kernel_size, dilation=1)

        # unfold
        x = self.unfold(x)  # (B, Cin*kH*kW, L), where L is the numbers of patches
        x = x.unsqueeze(1)  # (B, 1, Cin*kH*kW, L)
        L = x.size(-1)
        L_sqrt = int(math.sqrt(L))

        # erosion
        weight = self.weight.view(self.out_channels, -1)  # (Cout, Cin*kH*kW)
        weight = weight.unsqueeze(0).unsqueeze(-1)  # (1, Cout, Cin*kH*kW, 1)

        if self.type == 'erosion2d':
            x = weight - x  # (B, Cout, Cin*kH*kW, L)
        elif self.type == 'dilation2d':
            x = weight + x  # (B, Cout, Cin*kH*kW, L)
        else:
            raise ValueError

        if not self.soft_max:
            x, _ = torch.max(x, dim=2, keepdim=False)  # (B, Cout, L)
        else:
            x = torch.logsumexp(x * self.beta, dim=2, keepdim=False) / self.beta  # (B, Cout, L)

        if self.type == 'erosion2d':
            x = -1 * x

        # instead of fold, we use view to avoid copy
        x = x.view(-1, self.out_channels, L_sqrt, L_sqrt)  # (B, Cout, L/2, L/2)

        return x


class Dilation2d(Morphology):
    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=20):
        super(Dilation2d, self).__init__(in_channels, out_channels, kernel_size, soft_max, beta, 'dilation2d')


class Erosion2d(Morphology):
    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=20):
        super(Erosion2d, self).__init__(in_channels, out_channels, kernel_size, soft_max, beta, 'erosion2d')


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2. / 9. / 64.)).clamp_(-0.025, 0.025)
        nn.init.constant(m.bias.data, 0.0)


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return (PSNR / Img.shape[0])


def batch_SSIM(img, imclean, data_range):
    Img = img.permute(0, 2, 3, 1).data.cpu().numpy().astype(np.float32)
    Iclean = imclean.permute(0, 2, 3, 1).data.cpu().numpy().astype(np.float32)
    # print(Img.shape)
    SSIM = 0
    for i in range(Img.shape[0]):
        SSIM += compare_ssim(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range, multichannel=True)
    return (SSIM / Img.shape[0])


def data_augmentation(image, mode):
    out = np.transpose(image, (1, 2, 0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2, 0, 1))