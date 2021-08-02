import torch

from gans_package.models import GAN_Generator, GAN_Discriminator
from gans_package.models import  DCGAN_Generator, DCGAN_Discriminator
from gans_package.models import SNGAN_Discriminator
from gans_package.models import StyleGAN_Generator, StyleGAN_Discriminator
from gans_package.models import SRGAN_Generator, SRGAN_Discriminator
from gans_package.models.Pix2PixHD import *

from Loss_F.loss import W_Crit_Loss


SEED = 0

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device being used: {}".format(DEVICE))

    # GAN

    g = GAN_Generator(10, 784, 128, 4).to(DEVICE)
    print(g)
    d = GAN_Discriminator(784, 512, 2).to(DEVICE)
    print(d)
    del g
    del d

    # DCGAN

    g = DCGAN_Generator(10, 1, 64).to(DEVICE)
    print(g)
    d = DCGAN_Discriminator(1, 16).to(DEVICE)
    print(d)
    del g
    del d

    # SNGAN

    g = SNGAN_Discriminator(1, 64).to(DEVICE)
    print(g)
    del g

    # StyleGAN

    g = StyleGAN_Generator(256, 3, 512, 128, 256, 512, synthesis_layers=8).to(DEVICE)
    print(g)
    d = StyleGAN_Discriminator(3, 16).to(DEVICE)
    print(d)
    del g
    del d

    # SRGAN

    g = SRGAN_Generator().to(DEVICE)
    print(g)
    d = SRGAN_Discriminator().to(DEVICE)
    print(d)
    del g
    del d

    # Pix2PixHD
    g = GlobalGenerator(3, 64, 3, 4, 9, 4).to(DEVICE)
    print(g)
    d = Discriminator(3, 64, 1).to(DEVICE)
    print(d)
    tensor = torch.rand((2, 3, 512, 512)).to(DEVICE)
    print("Generator Input Size: {}".format(tensor.size()))
    out = g(tensor)
    print("Generator Output Size: {}".format(out.size()))
    d_out = d(out)
    print("Discriminator Last Tensor Output Size: {}".format(d_out[-1].size()))
    del g
    del d
    del tensor
    del out

    print("Program has Ended")
