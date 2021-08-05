import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, latent_size=100):
        super(VAE, self).__init__()

        self.latent_size = latent_size

        self.l1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.l1b = nn.BatchNorm2d(32)
        self.l2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.l2b = nn.BatchNorm2d(64)
        self.l3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.l3b = nn.BatchNorm2d(128)
        self.l4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.l4b = nn.BatchNorm2d(256)

        self.l41 = nn.Linear(256 * 4 * 4, self.latent_size)
        self.l42 = nn.Linear(256 * 4 * 4, self.latent_size)

        self.f = nn.Linear(self.latent_size, 256 * 4 * 4)

        self.l5 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.l6 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.l7 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.l8 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1)

    def encoder(self, x_in):
        h = F.leaky_relu(self.l1b(self.l1(x_in)))
        h = F.leaky_relu(self.l2b(self.l2(h)))
        h = F.leaky_relu(self.l3b(self.l3(h)))
        h = F.leaky_relu(self.l4b(self.l4(h)))

        h = h.view(h.size(0), -1)

        return self.l41(h), self.l42(h)

    def decoder(self, z):
        z = self.f(z)
        z = z.view(-1, 256, 4, 4)

        z = F.leaky_relu(self.l5(z))
        z = F.leaky_relu(self.l6(z))
        z = F.leaky_relu(self.l7(z))
        z = torch.sigmoid(self.l8(z))

        return z

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return torch.add(eps.mul(std), mu)

    def forward(self, x_in):
        mu, log_var = self.encoder(x_in)
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var
