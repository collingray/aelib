import gc
from multiprocessing import Pool

import torch
from tqdm.autonotebook import tqdm
from transformer_lens import HookedTransformer

from ..utils import truncate_seq


class ActivationsBuffer:
    """
    A data buffer to generate, shuffle, and store model activations.g
    """

    def __init__(
        self,
        model: str | HookedTransformer,
        dataset,
        layers: int | list[int],
        act_sites: str | list[str],
        buffer_size=256,
        min_capacity=128,
        model_batch_size=8,
        samples_per_seq=None,
        max_seq_length=None,
        act_size=None,
        shuffle_buffer=True,
        seed=None,
        device="cuda",
        dtype=torch.bfloat16,
        buffer_device="cpu",
        offload_device=None,
        refresh_progress=False,
    ):
        """
        :param model: the hf model name, or a preloaded transformer lens model
        :param dataset: the dataset to use
        :param layers: which layer(s) to get activations from
        :param act_sites: the tl site(s) to get activations from
        :param buffer_size: the size of the buffer, in number of activations
        :param min_capacity: the minimum guaranteed capacity of the buffer, in number of activations, used to determine when to refresh the buffer
        :param model_batch_size: the batch size to use in the model when generating activations
        :param samples_per_seq: the number of activations to randomly sample from each sequence. If None, all activations will be used
        :param max_seq_length: the maximum sequence length to use when generating activations. If None, the sequences will not be truncated
        :param act_size: the size of the activations vectors. If None, it will guess the size from the model's cfg
        :param shuffle_buffer: if True, the buffer will be shuffled after each refresh
        :param seed: the seed to use for dataset shuffling and activation sampling
        :param device: the device to use for the model, and for returned activations
        :param dtype: the dtype to use for the buffer and model
        :param buffer_device: the device to use for the buffer. If None, it will use the same device as the model
        :param offload_device: the device to offload the model to when not generating activations. If None, offloading is disabled. If using this, make sure to
            use a large enough buffer to avoid frequent offloading
        :param refresh_progress: If True, a progress bar will be displayed when refreshing the buffer
        """

        assert (
            isinstance(layers, list) and len(layers) > 0
        ), "layers must be a non-empty list of ints"

        self.act_names = [
            f"blocks.{layer}.{site}" for layer in layers for site in act_sites
        ]
        self.buffer_size = buffer_size
        self.min_capacity = min_capacity
        self.samples_per_seq = samples_per_seq
        self.max_seq_length = max_seq_length
        self.act_size = act_size
        self.shuffle_buffer = shuffle_buffer
        self.device = device
        self.dtype = dtype
        self.buffer_device = buffer_device or device
        self.offload_device = offload_device
        self.refresh_progress = refresh_progress
        self.final_layer = max(layers)  # the final layer that needs to be run
        self.n_layers = len(layers)

        if seed:
            torch.manual_seed(seed)

        # pointer to the current position in the dataset
        self.dataset_pointer = 0

        self.data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=model_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=8,
        )
        self.data_generator = iter(self.data_loader)

        if isinstance(model, str):
            self.model = HookedTransformer.from_pretrained_no_processing(
                model_name=model, device=device, dtype=dtype
            )
        else:
            self.model = model

        # if the act_size is not provided, use the size from the model's cfg
        if self.act_size is None:
            for site in act_sites:
                if site[:3] == "mlp":
                    if self.act_size is not None:
                        assert (
                            self.model.cfg.d_mlp == self.act_size
                        ), f"All act_sites must be of the same size"

                    self.act_size = self.model.cfg.d_mlp
                elif site[-3:] == "out":
                    if self.act_size is not None:
                        assert (
                            self.model.cfg.d_model == self.act_size
                        ), f"All act_sites must be of the same size"

                    self.act_size = self.model.cfg.d_model
                else:
                    raise ValueError(
                        f"Cannot determine act_size from act_site '{site}', please provide it manually"
                    )

        # if the buffer is on the cpu, pin it to memory for faster transfer to the gpu
        pin_memory = buffer_device == "cpu" and device == "cuda"

        # the buffer to store activations in, with shape (buffer_size, len(layers), act_size)
        self.buffer = torch.zeros(
            (buffer_size, len(layers), len(act_sites), self.act_size),
            dtype=dtype,
            pin_memory=pin_memory,
            device=buffer_device,
        )

        # pointer to read/write location in the buffer, reset to 0 after refresh is called
        # starts at buffer_size to be fully filled on first refresh
        self.buffer_pointer = self.buffer_size

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
        if self.offload_device:
            self.model.to(self.device)

        # start a progress bar if `refresh_progress` is enabled
        if self.refresh_progress:
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

            if self.max_seq_length:
                with Pool(8) as p:
                    seqs = p.starmap(
                        truncate_seq, [(seq, self.max_seq_length) for seq in seqs]
                    )

            # run the seqs through the model to get the activations
            out, cache = self.model.run_with_cache(
                seqs, stop_at_layer=self.final_layer + 1, names_filter=self.act_names
            )

            # clean up logits in order to free the graph memory
            del out
            self.empty_cache()

            # (batch, pos, layers*sites, act_size) -> (batch*samples_per_seq, layers*sites, act_size)
            acts = torch.stack([cache[name] for name in self.act_names], dim=-2)
            if self.samples_per_seq:
                acts = acts[
                    :, torch.randperm(acts.shape[-3])[: self.samples_per_seq]
                ].flatten(0, 1)
            else:
                acts = acts.flatten(0, 1)

            # (batch*samples_per_seq, layers*sites, act_size) -> (batch*samples_per_seq, layers, sites, act_size)
            acts = acts.view(
                acts.shape[0],
                self.n_layers,
                -1,
                self.act_size,
            )

            write_pointer = self.buffer_size - self.buffer_pointer

            new_acts = min(
                acts.shape[0], self.buffer_pointer
            )  # the number of acts to write, capped by buffer_pointer
            self.buffer[write_pointer : write_pointer + acts.shape[0]].copy_(
                acts[:new_acts], non_blocking=True
            )
            del acts

            # update the buffer pointer by the number of activations we just added
            self.buffer_pointer -= new_acts

            # update the progress bar
            if self.refresh_progress:
                pbar.update(new_acts)

        # close the progress bar
        if self.refresh_progress:
            pbar.close()

        # sync the buffer to ensure async copies are complete
        self.synchronize()

        # if shuffle_buffer is enabled, shuffle the buffer
        if self.shuffle_buffer:
            self.buffer = self.buffer[torch.randperm(self.buffer_size)]

        # if offloading is enabled, move the model back to `cfg.offload_device`, and clear the cache
        if self.offload_device:
            self.model.to(self.offload_device)
            self.empty_cache()

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
            out = self.buffer[self.buffer_pointer : self.buffer_pointer + batch]

        self.buffer_pointer += batch or 1

        return out

    def reset_dataset(self):
        """
        Reset the buffer to the beginning of the dataset without reshuffling.
        """
        self.data_generator = iter(self.data_loader)

    def will_refresh(self, batch: int = None):
        return (
            self.buffer_size - (self.buffer_pointer + (batch or 1)) < self.min_capacity
        )

    def empty_cache(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()

        if self.device == "mps":
            torch.mps.empty_cache()

    def synchronize(self):
        if self.device == "cuda":
            torch.cuda.synchronize()

        if self.device == "mps":
            torch.mps.synchronize()
