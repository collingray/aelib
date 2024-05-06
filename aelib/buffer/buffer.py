import torch
import datasets
from transformer_lens import HookedTransformer
from tqdm.autonotebook import tqdm
from multiprocessing import Pool
import gc

from ..utils import truncate_seq


class ActivationsBufferConfig:
    def __init__(
            self,
            model_name,
            layers,
            dataset_name,
            act_site,
            dataset_split=None,
            dataset_config=None,
            buffer_size=256,
            min_capacity=128,
            model_batch_size=8,
            samples_per_seq=None,
            max_seq_length=None,
            act_size=None,
            shuffle_buffer=False,
            seed=None,
            device="cuda",
            dtype=torch.bfloat16,
            buffer_device=None,
            offload_device=None,
            refresh_progress=False,
    ):
        """
        :param model_name: the hf model name
        :param layers: which layers to get activations from, passed as a list of ints
        :param dataset_name: the name of the hf dataset to use
        :param act_site: the tl key to get activations from
        :param dataset_split: the split of the dataset to use
        :param dataset_config: the config to use when loading the dataset
        :param buffer_size: the size of the buffer, in number of activations
        :param min_capacity: the minimum guaranteed capacity of the buffer, in number of activations, used to determine
        when to refresh the buffer
        :param model_batch_size: the batch size to use in the model when generating activations
        :param samples_per_seq: the number of activations to randomly sample from each sequence. If None, all
        activations will be used
        :param max_seq_length: the maximum sequence length to use when generating activations. If None, the sequences
        will not be truncated
        :param act_size: the size of the activations vectors. If None, it will guess the size from the model's cfg
        :param shuffle_buffer: if True, the buffer will be shuffled after each refresh
        :param seed: the seed to use for dataset shuffling and activation sampling
        :param device: the device to use for the model
        :param dtype: the dtype to use for the buffer and model
        :param buffer_device: the device to use for the buffer. If None, it will use the same device as the model
        :param offload_device: the device to offload the model to when not generating activations. If None, offloading
        is disabled. If using this, make sure to use a large enough buffer to avoid frequent offloading
        :param refresh_progress: If True, a progress bar will be displayed when refreshing the buffer
        """

        assert isinstance(layers, list) and len(layers) > 0, "layers must be a non-empty list of ints"

        self.model_name = model_name
        self.layers = layers
        self.dataset_name = dataset_name
        self.act_site = act_site
        self.act_names = [f"blocks.{layer}.{act_site}" for layer in layers]  # the tl keys to grab activations from todo
        self.dataset_split = dataset_split
        self.dataset_config = dataset_config
        self.buffer_size = buffer_size
        self.min_capacity = min_capacity
        self.model_batch_size = model_batch_size
        self.samples_per_seq = samples_per_seq
        self.max_seq_length = max_seq_length
        self.act_size = act_size
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.device = device
        self.dtype = dtype
        self.buffer_device = buffer_device or device
        self.offload_device = offload_device
        self.refresh_progress = refresh_progress
        self.final_layer = max(layers)  # the final layer that needs to be run


class ActivationsBuffer:
    """
    A data buffer to store MLP activations for training the autoencoder.

    Adapted from code by Neel Nanda: https://github.com/neelnanda-io/1L-Sparse-Autoencoder/blob/main/utils.py

    Cleaned up and modified to move all state inside the class, including the model reference, in order to simplify
    composition. The original design which reloads the buffer once it dips below 1/2 of capacity remains, although I'm
    unclear why this is done, probably just a hack to avoid having to worry about overflows
    """

    def __init__(self, cfg: ActivationsBufferConfig, hf_model=None):
        self.cfg = cfg

        if cfg.seed:
            torch.manual_seed(cfg.seed)

        # pointer to the current position in the dataset
        self.dataset_pointer = 0

        # load the dataset into a looping data loader
        if cfg.dataset_config:
            dataset = datasets.load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.dataset_split)
        else:
            dataset = datasets.load_dataset(cfg.dataset_name, split=cfg.dataset_split)

        self.data_loader = torch.utils.data.DataLoader(
            dataset['text'],
            batch_size=cfg.model_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=8
        )
        self.data_generator = iter(self.data_loader)

        # load the model into a HookedTransformer
        self.model = HookedTransformer.from_pretrained_no_processing(
            model_name=cfg.model_name,
            hf_model=hf_model,
            device=cfg.device,
            dtype=cfg.dtype
        )

        # if the act_size is not provided, use the size from the model's cfg
        if cfg.act_size is None:
            if cfg.act_site[:3] == "mlp":
                self.cfg.act_size = self.model.cfg.d_mlp
            elif cfg.act_site == "hook_mlp_out":
                self.cfg.act_size = self.model.cfg.d_model
            else:
                raise ValueError(f"Cannot determine act_size from act_site {cfg.act_site}, please provide it manually")

        # if the buffer is on the cpu, pin it to memory for faster transfer to the gpu
        pin_memory = cfg.buffer_device == "cpu"

        # the buffer to store activations in, with shape (buffer_size, len(layers), act_size)
        self.buffer = torch.zeros(
            (cfg.buffer_size, len(self.cfg.layers), cfg.act_size),
            dtype=cfg.dtype,
            pin_memory=pin_memory,
            device=cfg.buffer_device
        )

        # pointer to read/write location in the buffer, reset to 0 after refresh is called
        # starts at buffer_size to be fully filled on first refresh
        self.buffer_pointer = self.cfg.buffer_size

        # initial buffer fill
        self.refresh()

    @torch.no_grad()
    def refresh(self):
        """
        Whenever the buffer is refreshed, we remove the first `buffer_pointer` activations that were used, shift the
        remaining activations to the start of the buffer, and then fill the rest of the buffer with `buffer_pointer` new
        activations from the model.
        """

        # shift the remaining activations to the start of the buffer
        self.buffer = torch.roll(self.buffer, -self.buffer_pointer, 0)

        # if offloading is enabled, move the model to `cfg.device` before generating activations
        if self.cfg.offload_device:
            self.model.to(self.cfg.device)

        # start a progress bar if `refresh_progress` is enabled
        if self.cfg.refresh_progress:
            pbar = tqdm(total=self.buffer_pointer)

        # fill the rest of the buffer with `buffer_pointer` new activations from the model
        while self.buffer_pointer > 0:
            # get the next batch of seqs
            try:
                seqs = next(self.data_generator)
            except StopIteration:
                print("Data generator exhausted, resetting...")
                self.reset_dataset()
                seqs = next(self.data_generator)

            if self.cfg.max_seq_length:
                with Pool(8) as p:
                    seqs = p.starmap(truncate_seq, [(seq, self.cfg.max_seq_length) for seq in seqs])

            # run the seqs through the model to get the activations
            out, cache = self.model.run_with_cache(seqs, stop_at_layer=self.cfg.final_layer + 1,
                                                   names_filter=self.cfg.act_names)

            # clean up logits in order to free the graph memory
            del out
            torch.cuda.empty_cache()

            # store the activations in the buffer
            acts = torch.stack([cache[name] for name in self.cfg.act_names], dim=-2)
            # (batch, pos, layers, act_size) -> (batch*samples_per_seq, layers, act_size)
            if self.cfg.samples_per_seq:
                acts = acts[:, torch.randperm(acts.shape[-3])[:self.cfg.samples_per_seq]].flatten(0, 1)
            else:
                acts = acts.flatten(0, 1)

            write_pointer = self.cfg.buffer_size - self.buffer_pointer

            new_acts = min(acts.shape[0], self.buffer_pointer)  # the number of acts to write, capped by buffer_pointer
            self.buffer[write_pointer:write_pointer + acts.shape[0]].copy_(acts[:new_acts], non_blocking=True)
            del acts

            # update the buffer pointer by the number of activations we just added
            self.buffer_pointer -= new_acts

            # update the progress bar
            if self.cfg.refresh_progress:
                pbar.update(new_acts)

        # close the progress bar
        if self.cfg.refresh_progress:
            pbar.close()

        # sync the buffer to ensure async copies are complete
        torch.cuda.synchronize()

        # if shuffle_buffer is enabled, shuffle the buffer
        if self.cfg.shuffle_buffer:
            self.buffer = self.buffer[torch.randperm(self.cfg.buffer_size)]

        # if offloading is enabled, move the model back to `cfg.offload_device`, and clear the cache
        if self.cfg.offload_device:
            self.model.to(self.cfg.offload_device)
            torch.cuda.empty_cache()

        gc.collect()

        assert self.buffer_pointer == 0, "Buffer pointer should be 0 after refresh"

    @torch.no_grad()
    def next(self, batch: int = None):
        # if this batch read would take us below the min_capacity, refresh the buffer
        if self.will_refresh(batch):
            self.refresh()

        if batch is None:
            out = self.buffer[self.buffer_pointer]
        else:
            out = self.buffer[self.buffer_pointer:self.buffer_pointer + batch]

        self.buffer_pointer += batch or 1

        return out

    def reset_dataset(self):
        """
        Reset the buffer to the beginning of the dataset without reshuffling.
        """
        self.data_generator = iter(self.data_loader)

    def will_refresh(self, batch: int = None):
        return self.cfg.buffer_size - (self.buffer_pointer + (batch or 1)) < self.cfg.min_capacity
