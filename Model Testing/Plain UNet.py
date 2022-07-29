import os
import pandas as pd
import numpy as np
import albumentations as A
import cv2
from torch import nn
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

TRAIN_IMG_SIZE = (480, 640, 3)
VAL_IMG_SIZE = TRAIN_IMG_SIZE
TEST_IMG_SIZE = TRAIN_IMG_SIZE
N_CLASSES = 12
TRAIN_BATCH_SIZE = 4
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
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        skip_out = self.double_conv(x)
        down_out = self.down_sample(skip_out)
        return (down_out, skip_out)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_sample_mode):
        super(UpBlock, self).__init__()
        if up_sample_mode == 'conv_transpose':
            self.up_sample = nn.ConvTranspose2d(in_channels - out_channels, in_channels - out_channels, kernel_size=2,
                                                stride=2)
        elif up_sample_mode == 'bilinear':
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            raise ValueError("Unsupported `up_sample_mode` (can take one of `conv_transpose` or `bilinear`)")
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, out_classes=2, up_sample_mode='conv_transpose'):
        super(UNet, self).__init__()
        self.up_sample_mode = up_sample_mode
        # Downsampling Path
        self.down_conv1 = DownBlock(3, 64)
        self.down_conv2 = DownBlock(64, 128)
        self.down_conv3 = DownBlock(128, 256)
        self.down_conv4 = DownBlock(256, 512)
        # Bottleneck
        self.double_conv = DoubleConv(512, 1024)
        # Upsampling Path
        self.up_conv4 = UpBlock(512 + 1024, 512, self.up_sample_mode)
        self.up_conv3 = UpBlock(256 + 512, 256, self.up_sample_mode)
        self.up_conv2 = UpBlock(128 + 256, 128, self.up_sample_mode)
        self.up_conv1 = UpBlock(128 + 64, 64, self.up_sample_mode)
        # Final Convolution
        self.conv_last = nn.Conv2d(64, out_classes, kernel_size=1)

    def forward(self, x):
        x, skip1_out = self.down_conv1(x)
        x, skip2_out = self.down_conv2(x)
        x, skip3_out = self.down_conv3(x)
        x, skip4_out = self.down_conv4(x)
        x = self.double_conv(x)
        x = self.up_conv4(x, skip4_out)
        x = self.up_conv3(x, skip3_out)
        x = self.up_conv2(x, skip2_out)
        x = self.up_conv1(x, skip1_out)
        x = self.conv_last(x)
        return x


model = UNet().to(DEVICE)

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
