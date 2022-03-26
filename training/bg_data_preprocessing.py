from tqdm import tqdm
from transformers import AutoTokenizer
from typing import List


def prepare_pretraining_data(
    tokenizer: AutoTokenizer, file_paths: List[str], max_length: int
) -> List[str]:
    lines = build_textline_list(file_paths)
    print("Packing sequences...")
    packed_sequences = pack_sequences(lines, tokenizer, max_length)
    return packed_sequences


def build_textline_list(file_paths: List[str]):
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

    return lines


def pack_sequences(texts: List[str], tokenizer: AutoTokenizer, max_length: int) -> List[str]:
    data = []
    concat_len = 0
    concat_string = ""
    i = 0

    for line in tqdm(texts):
        # first tokenize the current line
        encoding = tokenizer.encode_plus(line, truncation=True)
        tokenized_line = encoding["input_ids"]

        # then we'll try to add it to the current sequence we're packing
        if concat_len + len(tokenized_line) < max_length:
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