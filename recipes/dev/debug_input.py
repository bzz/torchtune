# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from functools import partial

from omegaconf import DictConfig
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import config, training, utils
from torchtune.config._utils import _get_component_from_path
from torchtune.data import CROSS_ENTROPY_IGNORE_IDX, padded_collate_packed


log = utils.get_logger("DEBUG")


class DebugInputRecipe:
    def __init__(self, cfg: DictConfig):
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(dtype=cfg.dtype, device=self._device)
        self.cfg = cfg
        training.set_seed(seed=cfg.seed)

    def setup(self, cfg: DictConfig):
        """Setup tokenizer and dataloader"""
        # Initialize tokenizer
        self._tokenizer = config.instantiate(cfg.tokenizer)

        # Setup dataloader similar to other recipes
        batch_size = cfg.get("batch_size", 2)
        collate_name = cfg.get("collate_fn", "torchtune.data.padded_collate_sft")
        collate_fn = _get_component_from_path(collate_name)

        ds = config.instantiate(cfg.dataset, self._tokenizer)
        packed = cfg.dataset.get("packed", False)

        sampler = DistributedSampler(
            ds,
            num_replicas=1,
            rank=0,
            shuffle=False,
            seed=0,
        )

        self._dataloader = DataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            collate_fn=(
                partial(
                    collate_fn,
                    padding_idx=self._tokenizer.pad_id,
                    ignore_idx=CROSS_ENTROPY_IGNORE_IDX,
                )
                if not packed
                else padded_collate_packed
            ),
        )

    def print_input(self, cfg: DictConfig):
        """Read input from dataset and print it without calling model"""
        print("Printing first batch of inputs:")

        for batch_idx, batch in enumerate(self._dataloader):
            utils.batch_to_device(batch, self._device)

            # Print each sequence in the batch
            for idx in range(batch["tokens"].size(0)):
                print("=" * 20 + f" Sample {idx + 1}:\n")
                print("Tokens:")
                print(self._tokenizer.decode(batch["tokens"][idx].tolist()))

                if "labels" in batch:
                    print("\nLabels:")
                    not_paded = batch["labels"][idx] != self._tokenizer.pad_id
                    valid_labels = batch["labels"][idx][not_paded]

                    not_masked = valid_labels != CROSS_ENTROPY_IGNORE_IDX
                    valid_labels = valid_labels[not_masked]
                    print(self._tokenizer.decode(valid_labels.tolist()))

            # Only print first batch
            break


@config.parse
def main(cfg: DictConfig) -> None:
    config.log_config(recipe_name="DebugInputRecipe", cfg=cfg)
    recipe = DebugInputRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.print_input(cfg=cfg)


if __name__ == "__main__":
    sys.exit(main())
