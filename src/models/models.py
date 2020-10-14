import torch
import torch.nn as nn
import torch.nn.functional as F
from fspool import FSPool


def make_model_with_FSPoool(arg_dict):
    """
    Instantiates an auto encoder model using FSPool for the latent space
    Arguments: a dictionary with the following keys
        encoder_width: the number of features for the encoder internal layers
        latent_size: the number of features in the latent space (this will
            be added by one so that the number of masked elements is included
            in the latent space)
        decoder_width: the number of features in the decoder internal layers
    Returns:
        pytorch model
    """
    encoder_width = arg_dict.encoder_width
    latent_size = arg_dict.latent_size + 1
    decoder_width = arg_dict.decoder_width

    if arg_dict.vae:
        encoder = FSEncoderSizedVAE(input_channels=5,  # (particle_id, e, pt, eta, phi)
                                    dim=encoder_width,
                                    output_channels=latent_size)
    else:
        encoder = FSEncoderSized(input_channels=5,  # (particle_id, e, pt, eta, phi)
                                 dim=encoder_width,
                                 output_channels=latent_size)
    decoder = MLPDecoder(input_channels=latent_size,
                         dim=decoder_width,
                         output_channels=4,     # four momentum
                         set_size=20,           # maximum of 20 particles
                         particle_types=9       # number of classes for classifier
                         )

    if arg_dict.vae:
        net = VAE(set_encoder=encoder, set_decoder=decoder)
    else:
        net = Net(set_encoder=encoder, set_decoder=decoder)

    return net


class Net(nn.Module):
    def __init__(self, set_encoder, set_decoder, input_encoder=None):
        """
        In the auto-encoder setting, don't pass an input_encoder because the target set and mask is
        assumed to be the input.
        In the general prediction setting, must pass all three.
        """
        super().__init__()
        self.set_encoder = set_encoder
        self.input_encoder = input_encoder
        self.set_decoder = set_decoder

        for m in self.modules():
            if (
                isinstance(m, nn.Linear)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Conv1d)
            ):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input, target_set, target_mask):
        if self.input_encoder is None:
            # auto-encoder, ignore input and use target set and mask as input instead
            latent_repr = self.set_encoder(target_set, target_mask)
            target_repr = latent_repr
        else:
            # set prediction, use proper input_encoder
            latent_repr = self.input_encoder(input)
            # note that target repr is only used for loss computation in training
            # during inference, knowledge about the target is not needed
            target_repr = self.set_encoder(target_set, target_mask)

        predicted_set = self.set_decoder(latent_repr)

        return predicted_set, (target_repr, latent_repr)

class VAE(nn.Module):
    def __init__(self, set_encoder, set_decoder):
        """
        TODO:
        """
        super().__init__()
        self.set_encoder = set_encoder
        self.set_decoder = set_decoder

        for m in self.modules():
            if (
                isinstance(m, nn.Linear)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Conv1d)
            ):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input, target_set, target_mask):
        latent_mu, latent_var, mask_info = self.set_encoder(target_set, target_mask)
        std = torch.exp(0.5 * latent_var)
        z = latent_mu + torch.randn_like(std) * std

        predicted_set = self.set_decoder(torch.cat([z, mask_info], dim=1))

        return predicted_set, (latent_mu, latent_var)


class FSEncoderSized(nn.Module):
    """ FSEncoder, but one feature in representation is forced to contain info about sum of masks """

    def __init__(self, input_channels, output_channels, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, dim, 1),
            nn.ReLU(),
            nn.Conv1d(dim, dim, 1),
            nn.ReLU(),
            nn.Conv1d(dim, dim, 1),
            nn.ReLU(),
            nn.Conv1d(dim, output_channels - 1, 1),
        )
        self.pool = FSPool(output_channels - 1, 20, relaxed=False)

    def forward(self, x, mask=None):
        mask = mask.unsqueeze(1)

        x = self.conv(x)
        x = x / x.size(2)  # normalise so that activations aren't too high with big sets
        x = x * mask  # mask invalid elements away
        x, _ = self.pool(x)
        # include mask information in representation
        x = torch.cat([x, mask.mean(dim=2) * 4], dim=1)
        return x


class FSEncoderSizedVAE(nn.Module):
    """
    FSEncoder, but one feature in representation is forced to contain info about sum of masks
    The output to the latent space contains two elements, the mean and the standard deviation
    """

    def __init__(self, input_channels, output_channels, dim):
        super().__init__()
        self.output_channels = output_channels
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, dim, 1),
            nn.ReLU(),
            nn.Conv1d(dim, dim, 1),
            nn.ReLU(),
            nn.Conv1d(dim, dim, 1),
            nn.ReLU(),
            nn.Conv1d(dim, (output_channels - 1) * 2, 1),
        )
        self.pool = FSPool((output_channels - 1) * 2, 20, relaxed=False)

    def forward(self, x, mask=None):
        mask = mask.unsqueeze(1)

        x = self.conv(x)
        x = x * mask  # mask invalid elements away
        x, _ = self.pool(x)
        # mean and variation
        lat = x.view(-1, (self.output_channels - 1), 2)
        latent_mu, latent_var = lat[:, :, 0], lat[:, :, 1]
        # include mask information in representation
        mask_info = mask.mean(dim=2).view(-1,1)
        return [latent_mu, latent_var, mask_info]


class MLPDecoder(nn.Module):
    def __init__(self, input_channels, output_channels, set_size, dim, particle_types):
        super().__init__()
        self.output_channels = output_channels
        self.set_size = set_size
        self.particle_types = particle_types
        self.linear1 = nn.Linear(input_channels, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.linear_fourvector = nn.Linear(dim, output_channels * set_size)
        self.linear_classification = nn.Linear(dim, set_size*particle_types)

    def forward(self, x):
#         x = x.view(x.size(0), -1)
        x1 = F.elu(self.linear1(x))
        x2 = F.elu(self.linear2(x1))
        vec = self.linear_fourvector(x2)
        vec = vec.view(vec.size(0), self.output_channels, self.set_size)

        particle = self.linear_classification(x2)
        particle = particle.view(particle.size(0), self.particle_types, self.set_size)

        return vec, particle
