import nltk
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# 1. Setup
nltk.download("punkt", quiet=True)

# 2. Load + clean dataset
def clean_dataset():
    dataset = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track")

    def clean_up(example):
        src = (example["radiology_report"] or "").strip()
        tgt = (example["layman_report"] or "").strip()
        return len(src) > 0 and len(tgt) > 0
    
    clean_dataset = dataset.filter(clean_up)
    
    subset_train = clean_dataset["train"].shuffle(seed=42).select(range(100000))
    subset_val   = clean_dataset["validation"].select(range(8000))
    
    return subset_train, subset_val

# 4. Preprocessing + Tokenization
def preprocess_dataset(subset_train, subset_val, tokenizer, max_input_len=256, max_target_len=128):
    PREFIX = "Summarize this radiology report for a layperson: "

    def preprocess_function(batch):
        inputs = [PREFIX + x for x in batch["radiology_report"]]
        model_inputs = tokenizer(
            inputs,
            max_length=max_input_len,
            truncation=True,
            padding="longest"
        )

        labels = tokenizer(
            text_target=batch["layman_report"],
            max_length=max_target_len,
            truncation=True,
            padding="longest",
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    cols_to_keep = ["input_ids", "attention_mask", "labels"]

    tokenized_train = subset_train.map(
        preprocess_function,
        batched=True,
        remove_columns=[c for c in subset_train.column_names if c not in cols_to_keep],
    )
    tokenized_val = subset_val.map(
        preprocess_function,
        batched=True,
        remove_columns=[c for c in subset_val.column_names if c not in cols_to_keep],
    )
    return tokenized_train, tokenized_val