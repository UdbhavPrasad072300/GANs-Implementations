import torch
from models.GAN import Generator, Discriminator

torch.manual_seed(0)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device being used: {}".format(device))

    # Model Parameters
    g = Generator(10, 784, 128, 4)
    d = Discriminator(784, 512, 2)
    print(g)
    print(d)

    print("Program has Ended")
