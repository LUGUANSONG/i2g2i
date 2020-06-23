import torch.nn as nn
from sg2im.layers import GlobalAvgPool, Flatten, get_activation, build_cnn


class Encoder(nn.Module):
  def __init__(self, arch, normalization='none', activation='relu',
               padding='same', pooling='avg', args=None):
    super(Encoder, self).__init__()
    print("i2g2i.combine_sg2im_neural_motifs.encoder.Encoder")
    self.object_noise_dim = args.object_noise_dim

    cnn_kwargs = {
      'arch': arch,
      'normalization': normalization,
      'activation': activation,
      'pooling': pooling,
      'padding': padding,
    }
    cnn, D = build_cnn(**cnn_kwargs)
    self.cnn = nn.Sequential(cnn, GlobalAvgPool(), nn.Linear(D, self.object_noise_dim * 2))

  def forward(self, x):
    if x.dim() == 3:
      x = x[:, None]

    noise = self.cnn(x)
    mu = noise[:, :self.object_noise_dim]
    logvar = noise[:, self.object_noise_dim:]

    return mu, logvar


class ImageEncoder(nn.Module):
  def __init__(self, arch, normalization='batch', activation='leakyrelu-0.2',
               padding='same', pooling='avg', args=None):
    super(ImageEncoder, self).__init__()
    print("i2g2i.combine_sg2im_neural_motifs.encoder.ImageEncoder")
    input_dim = 3
    arch = 'I%d,%s' % (input_dim, arch)
    cnn_kwargs = {
      'arch': arch,
      'normalization': normalization,
      'activation': activation,
      'pooling': pooling,
      'padding': padding,
    }
    self.layout_noise_dim = args.layout_noise_dim
    self.object_noise_dim = args.object_noise_dim
    assert not (self.layout_noise_dim > 0 and self.object_noise_dim > 0), "layout noise, object noise should not be applied at the same time"
    self.cnn, D = build_cnn(**cnn_kwargs)
    for i in range(len(self.cnn)):
      if isinstance(self.cnn[i], nn.Conv2d):
        origin_cnn = self.cnn[i]
        self.cnn[i] = nn.Conv2d(origin_cnn.in_channels, origin_cnn.out_channels, kernel_size=origin_cnn.kernel_size,
                                stride=origin_cnn.stride, padding=1)

  def forward(self, x):
    noise = self.cnn(x)
    mu = noise[:, :self.layout_noise_dim+self.object_noise_dim, :, :]
    logvar = noise[:, self.layout_noise_dim+self.object_noise_dim:, :, :]

    return mu, logvar
