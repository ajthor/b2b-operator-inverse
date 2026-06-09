"""Sampling helpers shared by step-budget training entrypoints."""

from __future__ import annotations


class RandomFunctionSampler:
    """Uniform random function sampler for step-budget training."""

    def __init__(self, dataset_size: int, num_samples: int, seed: int):
        self.dataset_size = dataset_size
        self.num_samples = num_samples
        self.seed = seed

    def __iter__(self):
        import torch

        generator = torch.Generator()
        generator.manual_seed(self.seed)
        for _ in range(self.num_samples):
            yield int(
                torch.randint(
                    low=0,
                    high=self.dataset_size,
                    size=(1,),
                    generator=generator,
                ).item()
            )

    def __len__(self):
        return self.num_samples
