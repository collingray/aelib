import gc
import json
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from ..utils import TiedLinear


@dataclass
class AutoEncoderConfig:
    """The config for the `AutoEncoder` class

    Args:
        n_dim: the dimension of the input
        m_dim: the dimension of the hidden layer
        tied: if True, the decoder weights are tied to the encoder weights
        seed: the seed to use for pytorch rng
        device: the device to use for the model
        dtype: the dtype to use for the model
        record_data: if True, a variety of data will be recorded during on forward passes, including neuron
        firing frequencies and the average FVU
        num_firing_buckets: the number of buckets to use for recording neuron firing
        firing_bucket_size: the size of each bucket for recording neuron firing
        name: the name to use when saving the model
        save_dir: the directory to save the model to
    """
    n_dim: int
    m_dim: int
    tied: bool = False
    seed: Optional[int] = None
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    record_data: bool = True
    num_firing_buckets: int = 16
    firing_bucket_size: int = 2 ** 18
    name: str = "autoencoder"
    save_dir: Optional[str] = None


# Custom JSON encoder and decoder for AutoEncoderConfig, as torch.dtype is not serializable by default
class AutoEncoderConfigEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, AutoEncoderConfig):
            o_dict = o.__dict__.copy()
            o_dict["dtype"] = o_dict["dtype"].__str__()[6:]
            return o_dict

        return json.JSONEncoder.default(self, o)


class AutoEncoderConfigDecoder(json.JSONDecoder):
    def __init__(self):
        super().__init__(object_hook=self.dict_to_object)

    @staticmethod
    def dict_to_object(d):
        if "dtype" in d:
            d["dtype"] = getattr(torch, d["dtype"])
        return AutoEncoderConfig(**d)


class AutoEncoder(nn.Module):
    """
    Autoencoder model with a single hidden layer
    """

    def __init__(self, cfg: AutoEncoderConfig):
        super().__init__()

        self.cfg = cfg

        if cfg.seed:
            torch.manual_seed(cfg.seed)

        # encoder linear layer, goes from the models embedding space to the hidden layer
        self.encoder = nn.Linear(cfg.n_dim, cfg.m_dim, bias=False, dtype=cfg.dtype)
        # bias applied after encoding
        self.encoder_bias = nn.Parameter(torch.zeros(cfg.m_dim, dtype=cfg.dtype))
        self.relu = nn.ReLU()
        # decoder linear layer, goes from the hidden layer back to models embeddings
        if cfg.tied:
            self.decoder = TiedLinear(self.encoder)  # tied weights, uses same dtype/device as encoder
        else:
            self.decoder = nn.Linear(cfg.m_dim, cfg.n_dim, bias=False, dtype=cfg.dtype)
        # bias applied after decoding
        self.decoder_bias = nn.Parameter(torch.zeros(cfg.n_dim, dtype=cfg.dtype))

        init_decoder_w = torch.rand(cfg.n_dim, cfg.m_dim, dtype=cfg.dtype)
        init_decoder_w = 0.1 * init_decoder_w / init_decoder_w.norm(dim=0, p=2)
        self.decoder.weight.data = init_decoder_w
        self.encoder.weight.data = self.decoder.weight.T

        self.to(cfg.device)

        if cfg.record_data:
            self.register_data_buffers(cfg)

    def forward(self, x, lamb: float, decoder_norm_scale=False, mean_over_batch=True):
        encoded = self.encode(x)
        reconstructed = self.decode(encoded)
        loss, l1, mse = self.loss(x, reconstructed, encoded, lamb, decoder_norm_scale, mean_over_batch=mean_over_batch)

        if self.cfg.record_data:
            mean_mse = mse if mean_over_batch else mse.mean(dim=0)
            self.record_fvu_data(x, mean_mse)

        return encoded, loss, l1, mse

    def encode(self, x, record=True):
        x = self.relu(self.encoder(x) + self.encoder_bias)

        if self.cfg.record_data and record:
            self.record_firing_data(x)

        return x

    def decode(self, x):
        return self.decoder(x) + self.decoder_bias

    def loss(self, x, x_out, latent, lamb, decoder_norm_scale=False, mean_over_batch=True):
        l1 = self.normalized_l1(x, latent, decoder_norm_scale=decoder_norm_scale, mean_over_batch=mean_over_batch)
        mse = self.normalized_reconstruction_mse(x, x_out, mean_over_batch=mean_over_batch)
        total = (lamb * l1) + mse

        return total, l1, mse

    # Mappings for using SAE-Vis
    @property
    def W_enc(self):
        return self.encoder.weight.T

    @property
    def W_dec(self):
        return self.decoder.weight.T

    @property
    def b_enc(self):
        return self.encoder_bias

    @property
    def b_dec(self):
        return self.decoder_bias

    @staticmethod
    def normalized_reconstruction_mse(x, recons, mean_over_batch=True):
        """
        The MSE between the input and its reconstruction, normalized by the mean square of the input, averaged over the
        batch.

        [n_dim]*2 -> []
        Or
        [batch, n_dim]*2 -> []
        Or
        [batch, layer, n_dim]*2 -> [layer]
        """
        mse = ((x - recons) ** 2).mean(dim=-1) / (x ** 2).mean(dim=-1)
        return mse.mean(dim=0) if mean_over_batch else mse

    def normalized_l1(self, x, latent, decoder_norm_scale, mean_over_batch):
        """
        The L1 norm of the latent representation, normalized by the L2 norm of the input, averaged over the batch.

        [n_dim], [m_dim] -> []
        Or
        [batch, n_dim], [batch, m_dim] -> []
        Or
        [batch, layer, n_dim], [batch, layer, m_dim] -> [layer]
        """
        if decoder_norm_scale:
            latent = latent * self.decoder.weight.norm(dim=0, p=2)

        l1 = latent.norm(dim=-1, p=1) / x.norm(dim=-1, p=2)
        return l1.mean(dim=0) if mean_over_batch else l1

    def resample_neurons(self, inputs, batch_size, optimizer):
        """
        Resample dead neurons according to Anthropic's method
        (see https://transformer-circuits.pub/2023/monosemantic-features#appendix-autoencoder-resampling)
        """

        with torch.no_grad():
            gc.collect()
            torch.cuda.empty_cache()

            dead_neuron_mask = self.neuron_firings.view(-1, self.cfg.m_dim).sum(dim=0) == 0
            dead_neuron_idxs = torch.where(dead_neuron_mask)[0]

            if len(dead_neuron_idxs) == 0:
                print("No dead neurons to resample")
                return

            print(f"Resampling {len(dead_neuron_idxs)} dead neurons")

            square_input_losses = torch.cat(
                [self.forward(inputs[i:i + batch_size], mean_over_batch=False)[1] for i in
                 range(0, len(inputs), batch_size)]
            ) ** 2

            inputs = inputs.view(-1, self.cfg.n_dim)
            square_input_losses = square_input_losses.view(-1)

            # Resample neurons with probability proportional to the square loss
            neuron_probs = square_input_losses / square_input_losses.sum()

            # Resample inputs that caused dead neurons to fire, normalize to unit l2
            sampled_inputs = nn.functional.normalize(inputs[torch.multinomial(neuron_probs, len(dead_neuron_idxs))],
                                                     dim=-1)

            # Calculate the average norm of the encoder weights for the live neurons
            avg_encoder_norm = self.encoder.weight[~dead_neuron_mask].norm(dim=1).mean()

            # Set the weights of the dead neurons to the resampled inputs, normalized to 20% of the average encoder norm
            self.encoder.weight[dead_neuron_idxs] = sampled_inputs * avg_encoder_norm * 0.2

            # Set the decoder weights of the dead neurons to the normed inputs
            self.decoder.weight[:, dead_neuron_idxs] = sampled_inputs.T

            # Set the biases of the dead neurons to 0
            self.encoder_bias[dead_neuron_idxs] = 0

            # Reset optimizer params for changed weights/biases
            optim_state = optimizer.state_dict()["state"]

            # bias
            optim_state[1]["exp_avg"][dead_neuron_idxs] = 0
            optim_state[1]["exp_avg_sq"][dead_neuron_idxs] = 0

            # encoder weights
            optim_state[2]["exp_avg"][dead_neuron_idxs] = 0
            optim_state[2]["exp_avg_sq"][dead_neuron_idxs] = 0

            # decoder weights
            optim_state[3]["exp_avg"][:, dead_neuron_idxs] = 0
            optim_state[3]["exp_avg_sq"][:, dead_neuron_idxs] = 0

    def register_data_buffers(self, cfg):
        # Bucketed rolling avg. for memory efficiency
        self.register_buffer("num_encodes", torch.zeros(cfg.num_firing_buckets, device=cfg.device, dtype=torch.int32),
                             persistent=False)
        self.register_buffer("neuron_firings",
                             torch.zeros(cfg.num_firing_buckets, cfg.m_dim, device=cfg.device, dtype=torch.int32),
                             persistent=False)

        self.register_buffer("num_forward_passes", torch.tensor(0, device=cfg.device, dtype=torch.int32),
                             persistent=False)
        self.register_buffer("mse_ema", torch.tensor(0.0, device=cfg.device), persistent=False)
        self.register_buffer("input_avg", torch.zeros(cfg.n_dim, device=cfg.device), persistent=False)
        self.register_buffer("input_var", torch.zeros(cfg.n_dim, device=cfg.device), persistent=False)

    @torch.no_grad()
    def record_firing_data(self, x: torch.Tensor):
        if self.num_encodes[0] >= self.cfg.firing_bucket_size:
            # If we've exceeded the bucket size, roll the data and reset the first bucket
            self.num_encodes = torch.roll(self.num_encodes, 1, 0)
            self.neuron_firings = torch.roll(self.neuron_firings, 1, 0)
            self.num_encodes[0] = 0
            self.neuron_firings[0] = 0

        self.num_encodes[0] += x.shape[0]
        self.neuron_firings[0] += (x > 0).sum(dim=0)

    @torch.no_grad()
    def record_fvu_data(self, x: torch.Tensor, mse: torch.Tensor):
        batch_size = x.shape[0]

        n = self.num_forward_passes

        # Exponential moving average of the MSE, effectively updated `batch_size` times
        alpha = 0.05
        self.mse_ema = ((1 - alpha) ** batch_size) * (self.mse_ema - mse) + mse

        # Update variance to include new inputs
        self.input_avg = ((n * self.input_avg) + x.sum(dim=0)) / (n + batch_size)
        self.input_var = ((n * self.input_var) + ((x - self.input_avg) ** 2).sum(dim=0)) / (n + batch_size)

        self.num_forward_passes += batch_size

    def get_data(self):
        """
        Get data on the firing of different neurons in the hidden layer
        :return: A tuple containing a tensor of the frequency with which each neuron fires and the average number of
        neurons that fired per pass
        """

        firings = self.neuron_firings.sum(dim=0).float()
        passes = self.num_encodes.sum().item()

        freqs = firings / passes
        avg_fired = firings.sum(dim=-1) / passes

        # Calculate the average FVU
        avg_fvu = self.mse_ema / self.input_var.mean()

        return freqs, avg_fired, avg_fvu

    def save(self, checkpoint):
        filename = f"{self.cfg.name}_{checkpoint}"
        torch.save(self.state_dict(), f"{self.cfg.save_dir}/{filename}.pt")
        with open(f"{self.cfg.save_dir}/{self.cfg.name}_cfg.json", "w") as f:
            json.dump(self.cfg, f, cls=AutoEncoderConfigEncoder)

    @classmethod
    def load(cls, name, checkpoint, save_dir="./weights"):
        filename = f"{save_dir}/{name}_{checkpoint}.pt"
        with open(f"{save_dir}/{name}_cfg.json", "r") as f:
            cfg = json.load(f, cls=AutoEncoderConfigDecoder)
        model = cls(cfg)
        model.load_state_dict(torch.load(filename))
        print(f"Loaded model from {filename}")
        return model
