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

from bg_data_preprocessing import prepare_pretraining_data
from dataset_classes import PretrainingDataset
from train_tokenizer import train_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--config_source", type=str, default="roberta-base")
    parser.add_argument("--data_dir", type=str, default="./bg_data")
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--eval_accumulation_steps", type=int, default=32)
    parser.add_argument("--eval_steps", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--logging_steps", type=int, default=1000)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--save_dir", type=str, default="./roberta-base-bulgarian")
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--scheduler", type=str, default="linear")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--vocab_size", type=int, default=52000)
    parser.add_argument("--vocab_min_frequency", type=int, default=2)
    parser.add_argument("--warmup", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.save_dir, model_max_length=args.max_length
        )
    except OSError:
        print(f"Pretrained tokenizer not found at {args.save_dir}. Training...")
        train_tokenizer(
            args.data_dir,
            args.save_dir,
            args.config_source,
            args.vocab_size,
            args.vocab_min_frequency,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.save_dir, model_max_length=args.max_length
        )

    data_paths = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir)]
    text_data = prepare_pretraining_data(data_paths, tokenizer, args.max_length)
    train_data, test_data = train_test_split(
        text_data, test_size=args.test_size, random_state=args.seed
    )
    train_dataset = PretrainingDataset(train_data, tokenizer, args.max_length)
    test_dataset = PretrainingDataset(test_data, tokenizer, args.max_length)
    config = AutoConfig.from_pretrained(args.config_source, vocab_size=args.vocab_size)
    model = AutoModelForMaskedLM(config=config)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability
    )
    training_args = TrainingArguments(
        dataloader_num_workers=args.num_workers,
        eval_accumulation_steps=args.eval_accumulation_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps",
        fp16=True,
        gradient_accumulation_steps=args.accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.scheduler,
        num_train_epochs=args.epochs,
        output_dir=args.save_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        report_to="none",
        save_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
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