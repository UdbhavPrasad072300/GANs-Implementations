import torch
from models.GAN import Generator


def generator_test(self):
    self.assertEqual(True, False)


if __name__ == '__main__':
    model = Generator(10, 784, 128, 1)
    print(model)
