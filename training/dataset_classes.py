import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List


class PretrainingDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: AutoTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = texts

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> torch.Tensor:
        text = self.examples[index]
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation="max_length",
            max_length=self.max_length,
        )
        return inputs
