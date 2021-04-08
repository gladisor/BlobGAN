from models import DCGAN
from torchvision import transforms
from pathlib import Path
import matplotlib.pyplot as plt

path = Path().absolute()
model_path = path / 'data' / 'saved_models'

transform = transforms.ToPILImage()

model = DCGAN.load_from_checkpoint(model_path / 'large_dataset_rgb.ckpt')

n_row = 4
n_col = 4

z = model.get_noise(n_row * n_col)

fig, ax = plt.subplots(n_row, n_col)
idx = 0
for i in range(n_row):
    for j in range(n_col):
        x = z[idx].unsqueeze(0)
        y = model.G(x).squeeze(0)
        y = y.permute(1, 2, 0).detach().numpy()
        ax[i, j].imshow(y)
        idx += 1

plt.show()

# for i in range(z.shape[0]):
#     x = z[i].unsqueeze(0)
#     y = model.G(x).squeeze(0)
#     # print(y.shape)
#     im = transform(y)
#     im.show()
