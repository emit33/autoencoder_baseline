import math
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, data_shape, encoder_widths, latent_dim):
        super().__init__()
        self.data_shape = data_shape
        self.dim_in = math.prod(data_shape)
        self.flatten_data: bool = len(self.data_shape) > 1
        self.layers = self._build_layers(self.dim_in, encoder_widths, latent_dim)

    def _build_layers(self, dim_in, encoder_widths, latent_dim):
        intermediate_layer_widths = (dim_in,) + encoder_widths

        layers = []
        for dim_in, dim_out in zip(
            intermediate_layer_widths[:-1], intermediate_layer_widths[1:]
        ):
            layers.append(nn.Linear(dim_in, dim_out))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(encoder_widths[-1], latent_dim))

        return nn.Sequential(*layers)

    def forward(self, x_in):
        if self.flatten_data:
            batch_size = x_in.shape[0]
            x_in = x_in.view(batch_size, -1)

        z = self.layers(x_in)
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dim, decoder_widths, data_shape):
        super().__init__()

        self.data_shape = data_shape
        self.dim_out = math.prod(data_shape)
        self.flatten_data: bool = len(self.data_shape) > 1
        self.layers = self._build_layers(latent_dim, decoder_widths, self.dim_out)

    def _build_layers(self, latent_dim, decoder_widths, dim_out):
        intermediate_layer_widths = (latent_dim,) + decoder_widths

        layers = []
        for intermediate_dim_in, intermediate_dim_out in zip(
            intermediate_layer_widths[:-1], intermediate_layer_widths[1:]
        ):
            layers.append(nn.Linear(intermediate_dim_in, intermediate_dim_out))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(decoder_widths[-1], dim_out))

        return nn.Sequential(*layers)

    def forward(self, z):
        x_hat = self.layers(z)

        # Unflatten if needed
        if self.flatten_data:
            batch_size = x_hat.shape[0]
            x_hat = x_hat.view(batch_size, *self.data_shape)
        return x_hat


class AutoEncoder(nn.Module):
    def __init__(self, data_shape, encoder_widths, latent_dim):
        super().__init__()

        self.data_shape = data_shape
        self.encoder_widths = encoder_widths
        self.latent_dim = latent_dim

        decoder_widths = self.encoder_widths[::-1]

        self.encoder = Encoder(data_shape, encoder_widths, latent_dim)
        self.decoder = Decoder(latent_dim, decoder_widths, data_shape)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)

        return z, x_hat
