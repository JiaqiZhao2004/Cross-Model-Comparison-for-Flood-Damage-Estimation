import os
import pandas as pd
import numpy as np
import albumentations as A
import cv2
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

TRAIN_IMG_SIZE = (480, 640, 3)
VAL_IMG_SIZE = TRAIN_IMG_SIZE
TEST_IMG_SIZE = TRAIN_IMG_SIZE
N_CLASSES = 12
TRAIN_BATCH_SIZE = 2
VAL_BATCH_SIZE = TRAIN_BATCH_SIZE
TEST_BATCH_SIZE = 1
NUM_EPOCHS = 30
TRAIN_NUM_WORKERS = 2
VAL_NUM_WORKERS = 2
TEST_NUM_WORKERS = 1
PIN_MEMORY = True
LEARNING_RATE = 0.001
DEVICE = 'cuda'
CHECKPOINT_PATH = ''

LOAD_MODEL = False
START_EPOCH = 1

torch.set_default_dtype(torch.float32)
torch.backends.cudnn.benchmark = True

# Train DataFrame
df = pd.DataFrame(columns=['folder_path', 'image_name', 'extension'])

for folder_number in tqdm(range(1, 11)):
    for img in os.listdir("RescueNet Dataset/rescuenet-train-{}-of-10/{}/".format(folder_number, folder_number)):
        info = {
            'folder_path': ["RescueNet Dataset/rescuenet-train-{}-of-10/{}/".format(folder_number, folder_number)],
            'image_name': [img[0:5]],
            'extension': ['jpg']}
        info_ = pd.DataFrame(data=info)
        df = pd.concat((df, info_))

for img in os.listdir("RescueNet Dataset/rescuenet-train-missed/content/allrest"):
    info = {
        'folder_path': ["RescueNet Dataset/rescuenet-train-missed/content/allrest/"],
        'image_name': [img[0:5]],
        'extension': ['jpg']}
    info_ = pd.DataFrame(data=info)
    df = pd.concat((df, info_))

df = df.drop_duplicates(subset=['image_name'])

imgs = df.iloc[:, 0] + df.iloc[:, 1] + '.jpg'
imgs = imgs.reset_index(drop=True)
labels = 'RescueNet Dataset/rescuenet-train-labels/train-label-img/' + df.iloc[:, 1] + '_lab.png'
labels = labels.reset_index(drop=True)

train_df = pd.concat((imgs, labels), axis=1)

# Validation DataFrame
df = pd.DataFrame(columns=['folder_path', 'image_name'])

for img in os.listdir("RescueNet Dataset/rescuenet-val/val/val-org-img/"):
    info = {
        'folder_path': ["RescueNet Dataset/rescuenet-val/val/val-org-img/"],
        'image_name': [img[0:5]]
    }
    info_ = pd.DataFrame(data=info)
    df = pd.concat((df, info_))

df = df.drop_duplicates(subset=['image_name'])

imgs = df.iloc[:, 0] + df.iloc[:, 1] + '.jpg'
imgs = imgs.reset_index(drop=True)
labels = 'RescueNet Dataset/rescuenet-val/val/val-label-img/' + df.iloc[:, 1] + '_lab.png'
labels = labels.reset_index(drop=True)

val_df = pd.concat((imgs, labels), axis=1)

# Test DataFrame
df = pd.DataFrame(columns=['folder_path', 'image_name'])

for img in os.listdir("RescueNet Dataset/rescuenet-test/test/test-org-img/"):
    info = {
        'folder_path': ["RescueNet Dataset/rescuenet-test/test/test-org-img/"],
        'image_name': [img[0:5]]
    }
    info_ = pd.DataFrame(data=info)
    df = pd.concat((df, info_))

df = df.drop_duplicates(subset=['image_name'])

imgs = df.iloc[:, 0] + df.iloc[:, 1] + '.jpg'
imgs = imgs.reset_index(drop=True)
labels = 'RescueNet Dataset/rescuenet-test/test/test-label-img/' + df.iloc[:, 1] + '_lab.png'
labels = labels.reset_index(drop=True)

test_df = pd.concat((imgs, labels), axis=1)


# Dataset & Dataloader
class RescueNetDataset(Dataset):
    def __init__(self, df, transforms):
        super(RescueNetDataset, self).__init__()
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path, mask_path = self.df.loc[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        transformed = self.transforms(image=img, mask=mask)
        img = transformed['image']
        mask = transformed['mask']

        img = img / 255
        img = img.astype('float32')

        img = np.transpose(img, (2, 0, 1))

        mask_stacked = np.array([mask == 0])
        for i in range(1, 12):
            mask_stacked = np.concatenate([mask_stacked, np.array([mask == i])])
        mask = mask_stacked.astype(int)
        mask = mask.astype('int64')

        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)

        return img, mask


train_transforms = A.Compose([
    A.RandomScale(scale_limit=0.1),
    A.Flip(),
    A.Rotate(limit=30),
    A.Resize(TRAIN_IMG_SIZE[0], TRAIN_IMG_SIZE[1]),
])

val_transforms = A.Compose([
    A.Resize(VAL_IMG_SIZE[0], VAL_IMG_SIZE[1]),
])

test_transforms = A.Compose([
    A.Resize(TEST_IMG_SIZE[0], TEST_IMG_SIZE[1]),
])

train_dataset = RescueNetDataset(df=train_df, transforms=train_transforms)
val_dataset = RescueNetDataset(df=val_df, transforms=val_transforms)
test_dataset = RescueNetDataset(df=test_df, transforms=test_transforms)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=TRAIN_NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    drop_last=True
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
    num_workers=VAL_NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    drop_last=True,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=TEST_BATCH_SIZE,
    shuffle=False,
    num_workers=TEST_NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    drop_last=True,
)

# Model
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.nn import Module, Conv2d, Parameter, Softmax
import torch.nn.functional as F
from torch.nn.functional import upsample, normalize, interpolate
import logging

BatchNorm = nn.BatchNorm2d

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, deep_base=True):
        super(ResNet, self).__init__()
        self.deep_base = deep_base
        if not self.deep_base:
            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = BatchNorm(64)
        else:
            self.inplanes = 128
            self.conv1 = conv3x3(3, 64, stride=2)
            self.bn1 = BatchNorm(64)
            self.conv2 = conv3x3(64, 64)
            self.bn2 = BatchNorm(64)
            self.conv3 = conv3x3(64, 128)
            self.bn3 = BatchNorm(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        #         self.avgpool = nn.AvgPool2d(7, stride=1)
        #         self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        if self.deep_base:
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #         x = self.avgpool(x)
        #         x = x.view(x.size(0), -1)
        #         x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        model_path = './initmodel/resnet50_v2.pth'
        model.load_state_dict(torch.load(model_path), strict=False)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    #         model_path = './initmodel/resnet101_v2.pth'
    #         model.load_state_dict(torch.load(model_path), strict=False)
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
        model_path = './initmodel/resnet152_v2.pth'
        model.load_state_dict(torch.load(model_path), strict=False)
    return model


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self):
        raise NotImplementedError

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        # self.logger.info(f'Nbr of trainable parameters: {nbr_params}')

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel, self).__str__() + f'\nNbr of trainable parameters: {nbr_params}'
        # return summary(self, input_shape=(2, 3, 224, 224))


class encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoder, self).__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, x):
        x = self.down_conv(x)
        x_pooled = self.pool(x)
        return x, x_pooled


class decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.up_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_copy, x, interpolate=True):
        x = self.up(x)
        if interpolate:
            # Iterpolating instead of padding gives better results
            x = F.interpolate(x, size=(x_copy.size(2), x_copy.size(3)),
                              mode="bilinear", align_corners=True)
        else:
            # Padding in case the incomping volumes are of different sizes
            diffY = x_copy.size()[2] - x.size()[2]
            diffX = x_copy.size()[3] - x.size()[3]
            x = F.pad(x, (diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2))
        # Concatenate
        x = torch.cat([x_copy, x], dim=1)
        x = self.up_conv(x)
        return x


class MUNet(BaseModel):
    def __init__(self, num_classes, in_channels=3, freeze_bn=False, **_):
        super(MUNet, self).__init__()
        self.down1 = encoder(in_channels, 64)
        self.down2 = encoder(64, 128)
        self.down3 = encoder(128, 256)
        # self.down4 = encoder(256, 512)
        self.middle_conv = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.up1 = decoder(512, 256)
        self.up2 = decoder(256, 128)
        self.up3 = decoder(128, 64)
        # self.up4 = decoder(128, 64)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self._initialize_weights()
        if freeze_bn: self.freeze_bn()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x):
        x1, x = self.down1(x)
        x2, x = self.down2(x)
        x3, x = self.down3(x)
        # x4, x = self.down4(x)
        x = self.middle_conv(x)
        x = self.up1(x3, x)
        x = self.up2(x2, x)
        x = self.up3(x1, x)
        # x = self.up4(x1, x)
        x = self.final_conv(x)
        return x

    def get_backbone_params(self):
        # There is no backbone for unet, all the parameters are trained from scratch
        return []

    def get_decoder_params(self):
        return self.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()


###########################################################################
# Created by: Tashnim Chowdhury
# Email:tchowdh1@umbc.edu
# Copyright (c) 2021
###########################################################################

up_kwargs = {'mode': 'bilinear', 'align_corners': True}


class BaseNet(nn.Module):
    def __init__(self, nclass, backbone='resnet50', aux=False, se_loss=False, dilated=True, norm_layer=None,
                 root='./pretrain_models',
                 multi_grid=False, multi_dilation=None):
        super(BaseNet, self).__init__()
        self.nclass = nclass
        self.aux = aux
        self.se_loss = se_loss
        #         self.mean = mean
        #         self.std = std
        #         self.base_size = base_size
        #         self.crop_size = crop_size
        # copying modules from pretrained models
        if backbone == 'resnet50':
            self.pretrained = resnet50(pretrained=False)
        elif backbone == 'resnet101':
            self.pretrained = resnet101(pretrained=False)
        elif backbone == 'resnet152':
            self.pretrained = resnet152(pretrained=False)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        # bilinear upsample options
        self._up_kwargs = up_kwargs

        self.layer0 = nn.Sequential(self.pretrained.conv1, self.pretrained.bn1, self.pretrained.relu,
                                    self.pretrained.conv2, self.pretrained.bn2, self.pretrained.relu,
                                    self.pretrained.conv3, self.pretrained.bn3, self.pretrained.relu,
                                    self.pretrained.maxpool)

    def base_forward(self, x):
        x = self.layer0(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)
        return c1, c2, c3, c4


#     def evaluate(self, x, target=None):
#         pred = self.forward(x)
#         if isinstance(pred, (tuple, list)):
#             pred = pred[0]
#         if target is None:
#             return pred
#         correct, labeled = batch_pix_accuracy(pred.data, target.data)
#         inter, union = batch_intersection_union(pred.data, target.data, self.nclass)
#         return correct, labeled, inter, union

class ReDNetPlus(BaseNet):
    def __init__(self, nclass=N_CLASSES, backbone='resnet50', aux=False, se_loss=False, norm_layer=nn.BatchNorm2d,
                 **kwargs):
        super(ReDNetPlus, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = REDNetHead(1024, nclass, norm_layer)
        self.head2 = REDNetHead(2048, nclass, norm_layer)

        self.upsample_11 = nn.Upsample(scale_factor=2, mode="bilinear")

        self.conv5c = nn.Sequential(nn.Conv2d(1024, 256, 3, padding=1, bias=False),
                                    norm_layer(256),
                                    nn.ReLU())

        self.conv5a = nn.Sequential(nn.Conv2d(512, 128, 3, padding=1, bias=False),
                                    norm_layer(128),
                                    nn.ReLU())

        self.sc = Attention_Mod(256)
        self.sa = Attention_Mod(128)

        self.conv52 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                    norm_layer(256),
                                    nn.ReLU())
        self.conv51 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1, bias=False),
                                    norm_layer(256),
                                    nn.ReLU())

        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(256, nclass, 1))
        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(256, nclass, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, nclass, 1))

        self.global_context = GlobalPooling(2048, 256, norm_layer, self._up_kwargs)
        self.munet = MUNet(num_classes=256)

    def forward(self, x):
        imsize = x.size()[2:]
        c1, c2, c3, c4 = self.base_forward(x)

        feat1 = self.conv5a(c2)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(c3)

        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        sc_conv = self.upsample_11(sc_conv)
        feat_sum = sa_conv + sc_conv
        feat_sum = upsample(feat_sum, imsize, **self._up_kwargs)

        ## mini UNet implementation
        munet_output = self.munet(x)
        munet_output = upsample(munet_output, imsize, **self._up_kwargs)

        final_output_tensor = torch.cat((feat_sum, munet_output), 1)
        final_output = self.conv8(final_output_tensor)

        return final_output


class GlobalPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(GlobalPooling, self).__init__()
        self._up_kwargs = up_kwargs
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 norm_layer(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        pool = self.gap(x)
        return interpolate(pool, (h, w), **self._up_kwargs)


class REDNetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(REDNetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.sa = Attention_Mod(inter_channels)
        # self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x1):
        feat1 = self.conv5a(x1)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        return sa_output


class Attention_Mod(Module):
    """ Attention Module """

    def __init__(self, in_dim):
        super(Attention_Mod, self).__init__()
        self.channel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            input:
                x : input feature maps (B x C x H x W)
            returns:
                out : attention value + input feature
                attention: B x (HxW) x (HxW)
        """

        m_batchsize, m_channel, m_height, m_width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, m_width * m_height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, m_width * m_height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, m_width * m_height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, m_channel, m_height, m_width)

        out = self.gamma * out + x
        return out


model = ReDNetPlus().to(DEVICE)

# Training Preparation
"""
class_list = {
    'Background':0,
    'Debris':1,
    'Water':2,
    'Building_No_Damage':3,
    'Building_Minor_Damage':4,
    'Building_Major_Damage':5,
    'Building_Total_Destruction':6,
    'Vehicle':7,
    'Road':8,
    'Tree':9,
    'Pool':10,
    'Sand':11
}
"""

loss_fn = smp.losses.dice.DiceLoss(mode='multilabel')

loss_fn.__name__ = 'Dice_loss'

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

metric = [smp.utils.metrics.IoU()]

train_one_epoch = smp.utils.train.TrainEpoch(
    model=model,
    loss=loss_fn,
    metrics=metric,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True
)

val_one_epoch = smp.utils.train.ValidEpoch(
    model=model,
    loss=loss_fn,
    metrics=metric,
    device=DEVICE,
    verbose=True
)


def load_checkpoint(path, model):
    print("=> Loading checkpoint")
    model.load_state_dict(torch.load(path))


# Training
if LOAD_MODEL:
    load_checkpoint(CHECKPOINT_PATH, model)

for epoch in range(START_EPOCH, START_EPOCH + NUM_EPOCHS):
    print("EPOCH ", epoch)
    train_logs = train_one_epoch.run(dataloader=train_loader)
    print(train_logs)
    val_logs = val_one_epoch.run(dataloader=val_loader)
    print(val_logs)
    torch.save(model.state_dict(), CHECKPOINT_PATH)


# Evaluation
def test_one_epoch():
    model.eval()
    with torch.no_grad():
        IoU_List = np.zeros(shape=12, dtype='float32')
        metric = smp.utils.metrics.IoU()
        loop = tqdm(test_loader)
        for batch_index, (imgs, masks) in enumerate(loop):
            preds = model(imgs.to(DEVICE))  # shape [8,12,h,w]
            # masks shape [8,12,h,w]
            masks = masks.permute(1, 0, 2, 3)  # shape [12,8,h,w]
            preds_argmax = torch.argmax(preds, axis=1)  # shape [8,h,w]
            preds_stacked = (preds_argmax == 0).unsqueeze(0)  # shape [1,8,h,w]
            for i in range(1, 12):
                preds_stacked = torch.cat((preds_stacked, (preds_argmax == i).unsqueeze(0)))
            preds_stacked = preds_stacked.to(dtype=torch.int8)  # shape [12,8,h,w]

            for i in range(12):
                preds_layer = preds_stacked[i]  # shape [8,h,w]
                masks_layer = masks[i]  # shape [8,h,w]
                preds_layer = preds_layer.flatten()  # shape [8*h*w]
                masks_layer = masks_layer.flatten()  # shape [8*h*w]
                layer_iou = metric(preds_layer.cpu(), masks_layer)  # one float number
                IoU_List[i] += layer_iou
            loop.set_postfix(
                background=IoU_List[0] / (batch_index + 1),
                debris=IoU_List[1] / (batch_index + 1),
                water=IoU_List[2] / (batch_index + 1),
                building_superficial_damage=IoU_List[3] / (batch_index + 1),
                building_medium_damage=IoU_List[4] / (batch_index + 1),
                building_major_damage=IoU_List[5] / (batch_index + 1),
                building_total_destruction=IoU_List[6] / (batch_index + 1),
                vehicle=IoU_List[7] / (batch_index + 1),
                road=IoU_List[8] / (batch_index + 1),
                pool=IoU_List[9] / (batch_index + 1),
                tree=IoU_List[10] / (batch_index + 1),
                sand=IoU_List[11] / (batch_index + 1),
                mean_IoU=IoU_List[1:12].mean() / (batch_index + 1),
            )

        IoU_List = IoU_List / len(test_loader)
        print("background IoU = {}".format(IoU_List[0]))
        print("debris IoU = {}".format(IoU_List[1]))
        print("water IoU = {}".format(IoU_List[2]))
        print("building-superficial-damage IoU = {}".format(IoU_List[3]))
        print("building-medium-damage IoU = {}".format(IoU_List[4]))
        print("building-major-damage IoU = {}".format(IoU_List[5]))
        print("building-total-destruction IoU = {}".format(IoU_List[6]))
        print("vehicle IoU = {}".format(IoU_List[7]))
        print("road IoU = {}".format(IoU_List[8]))
        print("tree IoU = {}".format(IoU_List[9]))
        print("pool IoU = {}".format(IoU_List[10]))
        print("sand IoU = {}".format(IoU_List[11]))
        print("Validation got mean IoU {}".format(IoU_List[1:12].mean()))


test_one_epoch()
