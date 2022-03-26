import argparse
import os
from tokenizers import ByteLevelBPETokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="./roberta-small-bg")
    parser.add_argument("--data_dir", type=str, default="./bg_data")
    parser.add_argument("--vocab_size", type=int, default=52000)
    parser.add_argument("--vocab_min_frequency", type=int, default=2)
    return parser.parse_args()


def train_tokenizer(
    data_dir: str,
    save_dir: str,
    vocab_size: int,
    vocab_min_frequency: int,
):
    data_paths = [os.path.join(args.data_dir, f) for f in os.listdir(data_dir)]
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=data_paths,
        vocab_size=vocab_size,
        min_frequency=vocab_min_frequency,
        show_progress=True,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    )
    tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    args = parse_args()
    train_tokenizer(
        args.data_dir,
        args.save_dir,
        args.vocab_size,
        args.vocab_min_frequency,
        args.max_length,
    )
