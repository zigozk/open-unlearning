from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from atomic_tofu.io import read_json, read_jsonl
from data.qa import QADataset
from data.utils import preprocess_chat_instance


class LocalJSONQADataset(QADataset):
    """Official-QA-compatible local JSONL dataset used by frozen metric views."""

    def __init__(self, path, template_args, tokenizer, question_key="question", answer_key="answer", max_length=512, predict_with_generate=False, **_):
        Dataset.__init__(self)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = [{**row, "index": index} for index, row in enumerate(read_jsonl(path))]
        self.fs_data = None
        self.template_args = template_args
        self.question_key = question_key
        self.answer_key = answer_key
        self.predict_with_generate = predict_with_generate


class AtomicTOFUUnlearnDataset(Dataset):
    """Request-aware view with independent global-retain and logical local K pools."""

    def __init__(self, source_path, bundle_path, requests_path, template_args, tokenizer, logical_k=8, seed=0, max_length=512, **_):
        self.source = {row["qa_id"]: row for row in read_jsonl(source_path)}
        self.bundle = read_json(bundle_path)
        requests = {row["request_id"]: row for row in read_jsonl(requests_path)}
        self.requests = {request_id: requests[request_id] for request_id in self.bundle["request_ids"]}
        self.forget_ids = list(self.bundle["forget_qa_ids"])
        forget_set = set(self.forget_ids)
        self.retain_ids = sorted(set(self.source) - forget_set)
        self.qa_to_requests = {qa_id: [] for qa_id in self.forget_ids}
        for request_id, request in self.requests.items():
            for qa_id in request["forget_qa_ids"]:
                if qa_id in self.qa_to_requests:
                    self.qa_to_requests[qa_id].append(request_id)
            protected = set(request["protected_train_qa_ids"])
            if not protected <= set(self.retain_ids):
                raise ValueError(f"{request_id}: protected pool is not a retain subset")
        if any(not request_ids for request_ids in self.qa_to_requests.values()):
            raise ValueError("Every bundle forget QA must map to at least one request")
        self.template_args = template_args
        self.tokenizer = tokenizer
        self.logical_k = int(logical_k)
        self.seed = int(seed)
        self.max_length = int(max_length)

    def __len__(self):
        return len(self.forget_ids)

    def _tokenize(self, qa_id):
        row = self.source[qa_id]
        item = preprocess_chat_instance(self.tokenizer, self.template_args, row["question"], row["answer"], self.max_length, False)
        item["index"] = torch.tensor(row["source_index"])
        return item

    def _protected_ids(self, qa_id, index):
        request_ids = sorted(self.qa_to_requests[qa_id])
        pools = [sorted(set(self.requests[request_id]["protected_train_qa_ids"])) for request_id in request_ids]
        if any(not pool for pool in pools):
            raise ValueError(f"{qa_id}: active request has empty protected pool")
        generator = torch.Generator().manual_seed(self.seed * 1_000_003 + index)
        selected = []
        selected_groups = []
        union_pool = set().union(*map(set, pools))
        target_count = min(self.logical_k, len(union_pool))
        cursor = 0
        while len(selected) < target_count:
            pool = pools[cursor % len(pools)]
            remaining = [qa_id for qa_id in pool if qa_id not in selected]
            if remaining:
                candidate = remaining[torch.randint(0, len(remaining), (1,), generator=generator).item()]
                selected.append(candidate)
                selected_groups.append(cursor % len(pools))
            cursor += 1
            if cursor > max(1, self.logical_k) * max(4, len(pools) * 4):
                break
        return selected, selected_groups

    def __getitem__(self, index):
        forget_id = self.forget_ids[index]
        generator = torch.Generator().manual_seed(self.seed * 2_000_003 + index)
        retain_id = self.retain_ids[torch.randint(0, len(self.retain_ids), (1,), generator=generator).item()]
        protected_ids, protected_groups = self._protected_ids(forget_id, index)
        return {
            "forget": self._tokenize(forget_id),
            "retain": self._tokenize(retain_id),
            "protected": [self._tokenize(qa_id) for qa_id in protected_ids],
            "protected_request_group": protected_groups,
            "logical_k": len(protected_ids),
            "logical_k_requested": self.logical_k,
            "protected_duplicate_rate": 0.0,
        }
