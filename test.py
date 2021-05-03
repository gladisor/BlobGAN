from models import DCGAN
from torchvision import transforms
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from utils import BlobData
from torch.distributions import Normal

path = Path().absolute()

model_path = path / 'wandb' / 'RGB' / 'files' / 'BlobGAN' / '298o8k9c' / 'checkpoints' / 'epoch=1_global_step=9600.ckpt'
# model_path = path / 'wandb' / 'grayscale' / 'files' / 'BlobGAN' / 'trytk8ib' / 'checkpoints' / 'epoch=1_global_step=9800.ckpt'
# model_path = path / 'data' / 'saved_models' / 'good_run_grey.ckpt'
labels = torch.load('labeled_latents/RGB/labels.pt')
z = torch.load('labeled_latents/RGB/z.pt')

transform = transforms.ToPILImage()
model = DCGAN.load_from_checkpoint(model_path)
model.eval()

a = z[torch.where(labels == 5)]
b = z[torch.where(labels == 3)]

num_imgs = min(a.shape[0], b.shape[0])
print(num_imgs)
idx_a = torch.randint(0, a.shape[0], (num_imgs,))
idx_b = torch.randint(0, b.shape[0], (num_imgs,))

a_vect = a[idx_a].mean(dim=0, keepdim=True)
b_vect = b[idx_b].mean(dim=0, keepdim=True)

c = a_vect - b_vect
std = 0.25
dist = Normal(c, std)

y = model.G(a_vect).squeeze(0)
y = y.permute(1, 2, 0).detach().numpy()
plt.imshow(y)
plt.show()

y = model.G(b_vect).squeeze(0)
y = y.permute(1, 2, 0).detach().numpy()
plt.imshow(y)
plt.show()


n_row, n_col = 2, 2

fig, ax = plt.subplots(n_row, n_col)
idx = 0
for i in range(n_row):
    for j in range(n_col):
        sample = dist.sample()
        y = model.G(sample).squeeze(0)
        y = y.permute(1, 2, 0).detach().numpy()
        ax[i, j].imshow(y)
plt.show()










# n_row = 3
# n_col = 3
#
# z = model.get_noise(n_row * n_col)

#++++++++++++++++++#
# Labeling latents #
#++++++++++++++++++#
# labels = []
# # for i in range(2):
# for i in range(z.shape[0]):
#     x = z[i].unsqueeze(0)
#     y = model.G(x).squeeze(0)
#     y = y.permute(1, 2, 0).detach().numpy()
#     plt.imshow(y)
#     plt.show()
#     num = input('Number of blobs: ')
#     labels.append(int(num))
#
# labels = torch.LongTensor(labels)
#
# torch.save(labels, 'labels.pt')
# torch.save(z, 'z.pt')

#+++++++++++++++#
# Plotting grid #
#+++++++++++++++#
# path = Path().resolve()
# mode = 'RGB'
# data_path = path / 'data' / mode / 'test'
# mode_channels = {'1':1, 'RGB':3}
# img_channels = mode_channels[mode]
#
# data = BlobData(data_path, img_channels)

# fig, ax = plt.subplots(n_row, n_col)
# idx = 0
# for i in range(n_row):
#     for j in range(n_col):
#         # x = z[idx].unsqueeze(0)
#         # y = model.G(x).squeeze(0)
#         # y = y.permute(1, 2, 0).detach().numpy()
#         # ax[i, j].imshow(y, cmap='gray')
#
#         x, y = data[idx]
#         x = x.permute(1, 2, 0).detach().numpy()
#         ax[i, j].imshow(x)
#         idx += 1
#
# plt.show()

#+++++++++++++++++++++++#
# Histogram frequencies #
#+++++++++++++++++++++++#
# import seaborn as sns
# labels = torch.load('labeled_latents/grayscale/labels.pt').numpy()

# labels = []
# for i in range(100):
#     _, y = data[i]
#     labels.append(y.item() + 1)
#
# print(min(labels), max(labels))
# plt.xlabel('Number of blobs')
# plt.ylabel('Count')
#
# plt.hist(labels, bins=max(labels))
# # sns.histplot(labels)
# plt.show()
