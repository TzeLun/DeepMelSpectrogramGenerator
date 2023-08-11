from torch import nn
import torch
import torch.nn.functional as F
import numpy as np


# Mapping of loss function using keys from 'config' dictionary
loss_fn = {
    'mean_square_error': nn.MSELoss(),
    'binary_cross_entropy': nn.BCELoss(),
    'cross_entropy': nn.CrossEntropyLoss(),
    'binary_cross_entropy_logits': nn.BCEWithLogitsLoss()
}


# custom weight initialization on Generator and Discriminator based on DCGAN paper.
# Basically : weights randomly initialized from std normal dist with mu = 0, sigma = 0.02
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# reparameterization of the latent space
def reparameterize(mu, var):
    std = torch.exp(0.5 * var)
    eps = torch.randn_like(std)
    return mu + eps * std


# Compute the KL divergence between the standard normal distribution and the latent distribution
def kl_divergence(mean, var):
    return -0.5 * torch.sum(1 + torch.log(var) - torch.pow(mean, 2) - var)


# Compute the exponentially weighted moving average of a series. It doesn't accept an iterables however
# beta is the weights, calculated based on the time period of N - points : beta = 2 / (N + 1)
# Default beta is set for a period of 10 points
def EWMA(current, previous, beta=(2/11.0)):
    if torch.is_tensor(previous):

        if current.shape[0] == previous.shape[0]:
            return torch.add(torch.mul(current, beta), torch.mul(previous, 1 - beta))
        else:
            return current

    else:
        return current


# Compute the mean feature matching for the loss functions:
def MeanFeatureMatching(f_xr, f_xp, criterion):
    f_xr_mean = torch.mean(f_xr, 1)
    f_xp_mean = torch.mean(f_xp, 1)
    return criterion(f_xr_mean, f_xp_mean)


# Dynamic log compression or spectral normalization for mel spectrogram. log(1 + gamma * y)
# Added clipping to avoid inf values
def log_range_compression(y, gamma=1.0, clip_val=1e-5):
    return torch.log(gamma * torch.clamp(y, min=clip_val))


# Dynamic log decompression for spectrogram or mel spectrogram
def log_range_decompression(y, gamma=1.0):
    return torch.exp(y) / gamma


# One hot encoding in pytorch
# num_class is a 1D array. It stores information of the max class for multiple categories
# Returns a list containing the encoded categories in the order of their indices
def one_hot(labels, num_class):
    one_hot_labels = []
    for idx, nc in enumerate(num_class):
        one_hot_labels.append(F.one_hot(labels[idx], num_classes=nc))
    return one_hot_labels


# Check discriminator overfitting, rt = sign(yr). "T.Karras et. al. 2020: ADA". rt = 1.0 means overfitted
# real is a flag to determine if the all the predicted outputs are supposedly real (1) or fake (0)
def rt_score(y, real=True):
    y_prob = F.sigmoid(y)
    real_labels = y_prob[y_prob > 0.5]
    if real:
        return float(real_labels.shape[0]) / float(y_prob.shape[0])
    else:
        return 1.0 - float(real_labels.shape[0]) / float(y_prob.shape[0])


# Calculate the gradient penalty for WGAN-GP training, Added cls for CGANs
def gradient_penalty(critic, real, fake, cls=None, device='cpu'):
    batch_size, channel, h, w = real.shape
    epsilon = torch.rand((batch_size, 1, 1, 1)).repeat(1, channel, h, w).to(device)
    interpolated = real * epsilon + fake * (1 - epsilon)

    if cls is not None:
        critic_val, _ = critic(interpolated, cls)
    else:
        critic_val, _ = critic(interpolated)

    gradient = torch.autograd.grad(
        inputs=interpolated,
        outputs=critic_val,
        grad_outputs=torch.ones_like(critic_val),
        create_graph=True,
        retain_graph=True
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty


# Slicing the mel spectrogram into strips
# n is the number of strip
def mel_spectrogram_to_strips(y, n=1):
    _, _, w = y.shape
    ws = int(w / n)
    strips = []
    for i in range(n):
        strips.append(y[:, :, (i * ws):((i + 1)*ws)])
    return torch.stack(strips, 0)


# Stitching the strips back into a mel spectrogram
# Accepts batches. s is the strips in the same format as the output of above function
def mel_spectrogram_from_strips(s):
    ms = None
    for _, strip in enumerate(s):
        if ms is None:
            ms = strip
        else:
            ms = torch.concat([ms, strip], len(strip.shape) - 1)
    return ms


def numpy_mel_spectrogram_from_strips(s):
    ms = None
    for _, strip in enumerate(s):
        if ms is None:
            ms = strip
        else:
            ms = np.concatenate((ms, strip), len(strip.shape) - 1)
    return ms
