from typing import List, Literal, Optional, Union

import torch
from overrides import overrides

from autoencoder import *


@dataclass
class AutoEncoderMultiLayerConfig(AutoEncoderConfig):
    """The config for the `AutoEncoderMultiLayer` class

    Args:
        act_norms: the norms to use for layer activation renormalization, or the number of layers
        act_renorm_type: the type of renormalization to use for layer activations, one of "linear", "sqrt",
            "log", "none". Activations are scaled by act_renorm_scale*(avg(norms)/norms[layer]), where norms are the
            result of the act_renorm_type applied to act_norms
        act_renorm_scale: a global scale to apply to all activations after renormalization
        """
    act_norms: Union[List[float], int] = 1
    act_renorm_type: Literal["linear", "sqrt", "log", "none"] = "linear"
    act_renorm_scale: float = 1.0
    name = "multilayer_autoencoder"


# Custom JSON encoder and decoder for AutoEncoderMultiLayerConfig, as torch.dtype is not serializable by default
class AutoEncoderMultiLayerConfigEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, AutoEncoderMultiLayerConfig):
            o_dict = o.__dict__.copy()
            o_dict["dtype"] = o_dict["dtype"].__str__()[6:]
            return o_dict

        return json.JSONEncoder.default(self, o)


class AutoEncoderMultiLayerConfigDecoder(json.JSONDecoder):
    def __init__(self):
        super().__init__(object_hook=self.dict_to_object)

    @staticmethod
    def dict_to_object(d):
        if "dtype" in d:
            d["dtype"] = getattr(torch, d["dtype"])
        return AutoEncoderMultiLayerConfig(**d)


class AutoEncoderMultiLayer(AutoEncoder):
    """
    Autoencoder model with a single hidden layer that can be trained on activations from multiple layers of a model
    """

    def __init__(self, cfg: AutoEncoderMultiLayerConfig):
        self.num_layers = len(cfg.act_norms) if isinstance(cfg.act_norms, list) else cfg.act_norms

        super().__init__(cfg)

        if cfg.act_renorm_type == "none" or cfg.act_norms is int:
            # no renormalization
            norms = torch.ones(self.num_layers)
        elif cfg.act_renorm_type == "linear":
            norms = torch.Tensor(cfg.act_norms)
        elif cfg.act_renorm_type == "sqrt":
            norms = torch.Tensor(cfg.act_norms).sqrt()
        elif cfg.act_renorm_type == "log":
            norms = torch.Tensor(cfg.act_norms).log()
        else:
            raise ValueError(f"Invalid act_renorm_type {cfg.act_renorm_type}")

        self.register_buffer("act_scales", (cfg.act_renorm_scale * norms.mean() / norms).to(cfg.device, cfg.dtype))

    @overrides
    def encode(self, x, record=True, layer: Optional[int] = None):
        # A layer should only be specified if x is a single layer
        if layer is not None:  # x: [batch_size, n_dim]
            x = x * self.act_scales[layer]
        else:  # x: [batch_size, num_layers, n_dim]
            x = torch.einsum("bln,l->bln", x, self.act_scales)

        return super().encode(x, record=record and (layer is None))

    @overrides
    def decode(self, x, layer: Optional[int] = None):
        # A layer should only be specified if x is a single layer
        if layer is not None:  # x: [batch_size, n_dim]
            x = x / self.act_scales[layer]
        else:  # x: [batch_size, num_layers, n_dim]
            x = torch.einsum("bln,l->bln", x, 1 / self.act_scales)

        return super().decode(x)

    def register_data_buffers(self, cfg):
        # Bucketed rolling avg. for memory efficiency
        self.register_buffer("num_encodes", torch.zeros(cfg.num_firing_buckets, device=cfg.device, dtype=torch.int32),
                             persistent=False)
        self.register_buffer("neuron_firings",
                             torch.zeros(cfg.num_firing_buckets, self.num_layers, cfg.m_dim, device=cfg.device,
                                         dtype=torch.int32),
                             persistent=False)

        self.register_buffer("num_forward_passes", torch.tensor(0, device=cfg.device, dtype=torch.int32),
                             persistent=False)
        self.register_buffer("mse_ema", torch.zeros(self.num_layers, device=cfg.device),
                             persistent=False)
        self.register_buffer("input_avg", torch.zeros(self.num_layers, cfg.n_dim, device=cfg.device),
                             persistent=False)
        self.register_buffer("input_var", torch.zeros(self.num_layers, cfg.n_dim, device=cfg.device),
                             persistent=False)

    def save(self, checkpoint):
        filename = f"{self.cfg.name}_{checkpoint}"
        torch.save(self.state_dict(), f"{self.cfg.save_dir}/{filename}.pt")
        with open(f"{self.cfg.save_dir}/{self.cfg.name}_cfg.json", "w") as f:
            json.dump(self.cfg, f, cls=AutoEncoderMultiLayerConfigEncoder)

    @classmethod
    def load(cls, name, checkpoint, save_dir="./weights"):
        filename = f"{save_dir}/{name}_{checkpoint}.pt"
        with open(f"{save_dir}/{name}_cfg.json", "r") as f:
            cfg = json.load(f, cls=AutoEncoderMultiLayerConfigDecoder)
        model = cls(cfg)
        model.load_state_dict(torch.load(filename))
        print(f"Loaded model from {filename}")
        return model
