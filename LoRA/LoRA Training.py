#!/usr/bin/env python3
"""
Fine‑tune Ollama’s Deepseek model with LoRA on a pre‑tokenized CSV.

CSV must have at least two columns:
  - input_ids: a Python list of ints (e.g. "[12,45,23,...]")
  - attention_mask: same format as input_ids
"""
import argparse
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training

class TokenizedCSVDataset(Dataset):
    def __init__(self, csv_path: str, start: int, end: int):
        df = pd.read_csv(csv_path)
        df = df.iloc[start:end].reset_index(drop=True)
        self.input_ids = df["input_ids"].apply(eval).tolist()
        self.attention_mask = df["attention_mask"].apply(eval).tolist()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "labels": torch.tensor(self.input_ids[idx], dtype=torch.long),
        }

def parse_args():
    p = argparse.ArgumentParser(description="LoRA‑fine‑tune Deepseek via Ollama + PyTorch")
    p.add_argument("--csv_path",   type=str, required=True, help="Path to your tokenized CSV")
    p.add_argument("--start_idx",  type=int, default=0,    help="Row index to start at (inclusive)")
    p.add_argument("--end_idx",    type=int, default=None, help="Row index to end at (exclusive)")
    p.add_argument("--base_model", type=str, required=True,
                   help="Local path to Ollama Deepseek model (e.g. ~/.ollama/models/deepseek)")
    p.add_argument("--output_dir", type=str, default="deepseek-lora",
                   help="Where to save the fine‑tuned LoRA weights")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs",     type=int, default=3)
    p.add_argument("--lr",         type=float, default=3e-4)
    return p.parse_args()

def main():
    args = parse_args()

    # 1) Load tokenizer & model (in 8‑bit) from Ollama local folder
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=True,
        device_map="auto",
        trust_remote_code=True,
    )
    # 2) Prepare for int8 + apply LoRA
    model = prepare_model_for_int8_training(model)
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # 3) Build dataset & trainer
    dataset = TokenizedCSVDataset(args.csv_path, args.start_idx, args.end_idx)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=50,
        save_steps=200,
        save_total_limit=2,
        fp16=True,
        optim="adamw_torch",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # 4) Train
    trainer.train()

    # 5) Save LoRA adapter weights
    model.save_pretrained(args.output_dir)
    print(f"Finished fine‑tuning. LoRA weights saved to {args.output_dir}")

if __name__ == "__main__":
    main()
