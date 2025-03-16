# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Mapping, Optional, Union

from torchtune.data._messages import Message
from torchtune.datasets._packed import PackedDataset
from torchtune.datasets._sft import SFTDataset
from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform


def instruct_md_template_dataset(
    tokenizer: ModelTokenizer,
    *,
    source: str,
    column_map: Optional[Dict[str, str]] = None,
    train_on_input: bool = True,
    prompt_template: Optional[str] = None,
    new_system_prompt: Optional[str] = None,
    packed: bool = False,
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    **load_dataset_kwargs: Dict[str, Any],
) -> Union[SFTDataset, PackedDataset]:
    if prompt_template is None:
        prompt_template = (  # alpaca style
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
        )

    class MarkdownToMessages(Transform):
        def __init__(
            self,
            template: str,
            train_on_input: bool = True,
            new_system_prompt: Optional[str] = None,
            column_map: Optional[Dict[str, str]] = None,
        ):
            self.template = template
            self.train_on_input = train_on_input
            self.column_map = column_map
            self.new_system_prompt = new_system_prompt

        def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
            column_map = self.column_map or {}
            k_instr = column_map.get("instruction", "instruction")
            k_input = column_map.get("input", "input")
            k_output = column_map.get("output", "output")
            messages = []
            if self.new_system_prompt is not None:
                messages.append(
                    Message(
                        role="system",
                        content=self.new_system_prompt,
                        masked=True,
                        eot=True,
                    )
                )
            messages.extend(
                [
                    Message(
                        role="user",
                        content=self.template.format(
                            **{"instruction": sample[k_instr], "input": sample[k_input]}
                        ),
                        masked=not self.train_on_input,
                        eot=True,
                    ),
                    Message(
                        role="assistant",
                        content=sample[k_output],
                        masked=False,
                        eot=True,
                    ),
                ]
            )
            return {"messages": messages}

    message_transform = MarkdownToMessages(
        prompt_template, train_on_input, new_system_prompt, column_map
    )
    ds = SFTDataset(
        source=source,
        message_transform=message_transform,
        model_transform=tokenizer,
        filter_fn=filter_fn,
        split=split,
        **load_dataset_kwargs,
    )
    if packed:
        if tokenizer.max_seq_len is None:
            raise ValueError(
                "PackedDataset requires a max_seq_len to be set on the tokenizer."
            )
        return PackedDataset(ds, max_seq_len=tokenizer.max_seq_len)
    return ds
