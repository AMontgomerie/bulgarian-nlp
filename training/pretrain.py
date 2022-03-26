import argparse
import os
from sklearn.model_selection import train_test_split
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
from bg_data_preprocessing import prepare_pretraining_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--accumulation_steps", type=int, default=1)
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
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--scheduler", type=str, default="linear")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--warmup", type=float, default=0.05)
    parser.add_argument("--test_size", type=float, default=0.1)
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
    text_data = prepare_pretraining_data(data_paths, tokenizer, args.max_length)
    train_data, test_data = train_test_split(
        text_data, test_size=args.test_size, random_state=args.seed
    )
    train_dataset = PretrainingDataset(train_data, tokenizer)
    test_dataset = PretrainingDataset(test_data, tokenizer)
    config = AutoConfig(
        vocab_size=args.vocab_size,
        max_position_embeddings=args.max_position_embeddings,
        num_attention_heads=args.num_attention_heads,
        num_hidden_layers=args.num_hidden_layers,
        type_vocab_size=args.type_vocab_size,
    )
    model = AutoModelForMaskedLM(config=config)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability
    )
    training_args = TrainingArguments(
        dataloader_num_workers=args.num_workers,
        fp16=True,
        gradient_accumulation_steps=args.accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.scheduler,
        num_train_epochs=args.epochs,
        output_dir=args.save_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        report_to="none",
        seed=args.seed,
        warmup_ratio=args.warmup,
        save_steps=args.save_steps,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    trainer.train()
    trainer.save_model(args.save_dir)