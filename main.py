import torch
from models.GAN import GAN_Generator, GAN_Discriminator
from models.DCGAN import  DCGAN_Generator, DCGAN_Discriminator
from models.SNGAN import SNGAN_Discriminator
from models.StyleGAN import StyleGAN_Generator, StyleGAN_Discriminator

from Loss_F.loss import W_Crit_Loss

torch.manual_seed(0)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device being used: {}".format(device))

    # GAN

    g = GAN_Generator(10, 784, 128, 4).to(device)
    print(g)
    d = GAN_Discriminator(784, 512, 2).to(device)
    print(d)
    del g
    del d

    # DCGAN

    g = DCGAN_Generator(10, 1, 64).to(device)
    print(g)
    d = DCGAN_Discriminator(1, 16).to(device)
    print(d)
    del g
    del d

    # SNGAN

    g = SNGAN_Discriminator(1, 64).to(device)
    print(g)
    del g

    # StyleGAN

    g = StyleGAN_Generator(256, 3, 512, 128, 256, 512, synthesis_layers=8).to(device)
    print(g)
    d = StyleGAN_Discriminator(3, 16).to(device)
    print(d)
    del g
    del d

    print("Program has Ended")
