import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List


class PretrainingDataset(Dataset):
    def __init__(
        self, tokenizer: AutoTokenizer, file_paths: List[str], max_length: int
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        lines = []
        i = 1
        for path in file_paths:
            print("Reading {}/{} {}".format(i, len(file_paths), path))
            with open(path, encoding="utf-8") as f:
                lines.extend(
                    [
                        line
                        for line in f.read().splitlines()
                        if (len(line) > 0 and not line.isspace())
                    ]
                )
            i += 1
        print("Packing sequences...")
        packed_sequences = self.pack_sequences(lines)
        self.examples = packed_sequences
        print("Dataset ready.")

    def pack_sequences(self, text: str) -> List[str]:
        data = []
        concat_len = 0
        concat_string = ""
        i = 0
        checkpoints = [i for i in range(10, 110, 10)]

        for line in text:
            percent = round((i / len(text)) * 100)
            if percent in checkpoints:
                print("{}% complete".format(percent))
                checkpoints.pop(0)

            # first tokenize the current line
            encoding = self.tokenizer.encode_plus(line, truncation=True)
            tokenized_line = encoding["input_ids"]

            # then we'll try to add it to the current sequence we're packing
            if concat_len + len(tokenized_line) < self.max_length:
                concat_len += len(tokenized_line)
                concat_string += line

            # if the current sequence is already full, add it and make a new one
            else:
                data.append(concat_string)
                concat_len = len(tokenized_line)
                concat_string = line

            i += 1

        # we'll have one unfinished sequence left over after iterating
        data.append(concat_string)

        return data

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> torch.Tensor:
        text = self.examples[index]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )
        return torch.tensor(encoding["input_ids"], dtype=torch.long)
