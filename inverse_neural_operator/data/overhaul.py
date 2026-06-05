"""Dataset loading for migrated overhaul training stages."""

from __future__ import annotations


def load_overhaul_dataset(config, split: str):
    if config.dataset.name == "fwi":
        from inverse_neural_operator.data.fwi_hf import load_fwi_dataset

        return load_fwi_dataset(
            split=split,
            source=config.dataset.source or "ajthor/fwi",
            sample_limit=config.dataset.sample_limit,
        )
    if config.dataset.name in {"burgers", "burgers_1d"}:
        from inverse_neural_operator.data.burgers_hf import load_burgers_dataset

        return load_burgers_dataset(
            split=split,
            source=config.dataset.source or "ajthor/burgers-fenics",
            sample_limit=config.dataset.sample_limit,
        )
    raise ValueError(
        f"Unsupported migrated dataset {config.dataset.name!r}. "
        "Supported values are: fwi, burgers, burgers_1d."
    )
