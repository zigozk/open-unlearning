import torch
import transformers
from typing import Dict, Sequence
from data.utils import IGNORE_INDEX


class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        padding_side: str = "right",
        index: str = None,
    ):
        self.tokenizer = tokenizer
        self.padding_side = padding_side
        self.index = index

    def get_instances_from_key(self, instances: Sequence[Dict], key: str):
        ret_instances = [instance[key] for instance in instances]
        return ret_instances

    def _pad_tokens(self, input_ids, padding_value):
        if self.padding_side == "right":
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=padding_value
            )
        else:
            input_ids = torch.nn.utils.rnn.pad_sequence(
                [torch.flip(i, dims=[0]) for i in input_ids],
                batch_first=True,
                padding_value=padding_value,
            ).flip(dims=[1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        assert isinstance(instances[0], dict)
        return_dct = {}
        if "input_ids" not in instances[0]:
            for key in instances[0].keys():
                key_instances = self.get_instances_from_key(
                    instances=instances, key=key
                )
                return_dct[key] = self(key_instances)
        else:
            input_ids = [instance["input_ids"] for instance in instances]
            input_ids = self._pad_tokens(input_ids, self.tokenizer.pad_token_id)
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
            return_dct.update({"input_ids": input_ids})
            return_dct.update({"attention_mask": attention_mask})
            if "labels" in instances[0]:
                labels = [instance["labels"] for instance in instances]
                labels = self._pad_tokens(labels, IGNORE_INDEX)
                return_dct.update({"labels": labels})
            if self.index:
                if self.index in instances[0]:
                    return_dct.update(
                        {
                            self.index: torch.tensor(
                                [example[self.index] for example in instances]
                            )
                        }
                    )
                else:
                    raise Warning(f"{self.index} not found in dataset")
        return return_dct


class AtomicTOFURequestCollator(DataCollatorForSupervisedDataset):
    """Collate request-local protected lists without treating logical K as a microbatch."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # configs/data wraps the sole atomic dataset once as the `forget` split.
        if "forget" in instances[0] and isinstance(instances[0]["forget"], dict) and "protected" in instances[0]["forget"]:
            instances = [instance["forget"] for instance in instances]
        if "protected" not in instances[0]:
            return super().__call__(instances)
        protected = [sample for instance in instances for sample in instance["protected"]]
        group_keys = [
            (owner, request_group)
            for owner, instance in enumerate(instances)
            for request_group in instance["protected_request_group"]
        ]
        group_index = {key: index for index, key in enumerate(dict.fromkeys(group_keys))}
        owners = [group_index[key] for key in group_keys]
        return {
            "forget": super().__call__([instance["forget"] for instance in instances]),
            "retain": super().__call__([instance["retain"] for instance in instances]),
            "protected": super().__call__(protected),
            "protected_owner": torch.tensor(owners, dtype=torch.long),
            "logical_k": torch.tensor([instance["logical_k"] for instance in instances], dtype=torch.long),
            "logical_k_requested": torch.tensor([instance["logical_k_requested"] for instance in instances], dtype=torch.long),
            "protected_duplicate_rate": torch.tensor([instance["protected_duplicate_rate"] for instance in instances], dtype=torch.float),
        }
