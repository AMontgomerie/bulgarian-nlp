import argparse
import os
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from train_tokenizer import train_tokenizer
from dataset_classes import PretrainingDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="./roberta-small-bg")
    parser.add_argument("--data_dir", type=str, default="./bg_data")
    parser.add_argument("--vocab_size", type=int, default=52000)
    parser.add_argument("--vocab_min_frequency", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_position_embeddings", type=int, default=514)
    parser.add_argument("--num_attention_heads", type=int, default=12)
    parser.add_argument("--num_hidden_layers", type=int, default=6)
    parser.add_argument("--type_vocab_size", type=int, default=1)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=2000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_tokenizer(
        args.data_dir, args.save_dir, args.vocab_size, args.vocab_min_frequency
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.save_dir, model_max_length=args.max_length
    )
    data_paths = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir)]
    train_dataset = PretrainingDataset(tokenizer, data_paths)
    eval_dataset = PretrainingDataset(tokenizer, ["bg_text_26000000.txt"])
    config = AutoConfig(
        vocab_size=args.vocab_size,
        max_position_embeddings=args.max_position_embeddings,
        num_attention_heads=args.num_attention_heads,
        num_hidden_layers=args.num_hidden_layers,
        type_vocab_size=args.type_vocab_size,
    )
    model = AutoModelForMaskedLM(config=config)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability
    )
    training_args = TrainingArguments(
        output_dir=args.save_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=True,
    )
    trainer.train()
    trainer.save_model(args.save_dir)