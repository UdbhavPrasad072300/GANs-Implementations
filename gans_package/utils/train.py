import torch
import torch.nn as nn


def Wasserstein_Crit_Train(critic, generator, critic_optimizer, critic_loss, critic_repeats, real, latent_size,
                           device="cpu"):
    b = real.size(0)
    mean_loss = 0

    for _ in range(critic_repeats):
        critic.zero_grad()
        noise = torch.randn(b, latent_size).to(device)
        fake = generator(noise)
        fake_pred = critic(fake.detach())
        real_pred = critic(real)

        epsilon = torch.randn(b, 1, 1, 1, requires_grad=True).to(device)
        mixed_images = real * epsilon + fake.detach() * (1 - epsilon)
        mixed_scores = critic(mixed_images)
        gradients = (torch.autograd.grad(
            inputs=mixed_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
        )[0])
        gradients = gradients.view(len(gradients), -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = torch.mean((gradient_norm - 1) ** 2)
        loss = critic_loss(fake_pred, real_pred, penalty)
        loss.backward()
        mean_loss += loss.item() / critic_repeats
        critic_optimizer.step()

    return mean_loss
