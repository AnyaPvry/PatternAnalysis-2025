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
