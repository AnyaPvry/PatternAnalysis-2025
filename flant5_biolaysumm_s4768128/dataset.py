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
def load_clean_dataset():
    dataset = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track")

    def clean_up(example):
        src = (example["radiology_report"] or "").strip()
        tgt = (example["layman_report"] or "").strip()
        return len(src) > 0 and len(tgt) > 0
    return dataset.filter(clean_up)

# 4. Preprocessing + Tokenization
def preprocess_dataset(dataset, tokenizer, max_input_len=256, max_target_len=128):
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

    tokenized_train = dataset["train"].map(
        preprocess_function,
        batched=True,
        remove_columns=[c for c in dataset["train"].column_names if c not in cols_to_keep],
    )
    tokenized_val = dataset["validation"].map(
        preprocess_function,
        batched=True,
        remove_columns=[c for c in dataset["validation"].column_names if c not in cols_to_keep],
    )
    return tokenized_train, tokenized_val