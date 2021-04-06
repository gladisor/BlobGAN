from models import DCGAN
from torchvision import transforms

transform = transforms.ToPILImage()

model = DCGAN.load_from_checkpoint('good_run_grey.ckpt')

z = model.get_noise(10)

for i in range(z.shape[0]):
    x = z[i].unsqueeze(0)
    y = model.G(x).squeeze(0)
    print(y.shape)
    # im = transform(y)
    # im.show()
