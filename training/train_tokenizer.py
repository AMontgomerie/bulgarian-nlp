import argparse
import os
from tokenizers import ByteLevelBPETokenizer
from transformers import AutoTokenizer, RobertaTokenizerFast, PreTrainedTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="./roberta-base-bulgarian")
    parser.add_argument("--data_dir", type=str, default="./bg_data")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--model_type", type=str, default="roberta-base")
    parser.add_argument("--vocab_size", type=int, default=52000)
    parser.add_argument("--vocab_min_frequency", type=int, default=2)
    return parser.parse_args()


def train_tokenizer(
    model_type: str,
    data_dir: str,
    save_dir: str,
    vocab_size: int,
    vocab_min_frequency: int,
    max_length: int,
):
    data_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=data_paths,
        vocab_size=vocab_size,
        min_frequency=vocab_min_frequency,
        show_progress=True,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    )
    # save vocab.txt and merges.txt
    tokenizer.save_model(save_dir)
    # get the config for the model type we're using
    tokenizer_class = get_tokenizer_type(model_type)
    # load the new vocab and merges with the model's tokenizer config and save
    tokenizer = tokenizer_class.from_pretrained(save_dir, model_max_length=max_length)
    tokenizer.save_pretrained(save_dir)


def get_tokenizer_type(model_name: str) -> PreTrainedTokenizer:
    tokenizer_class = AutoTokenizer

    if "roberta" in model_name:
        tokenizer_class = RobertaTokenizerFast

    return tokenizer_class


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    train_tokenizer(
        args.model_type,
        args.data_dir,
        args.save_dir,
        args.vocab_size,
        args.vocab_min_frequency,
        args.max_length,
    )
