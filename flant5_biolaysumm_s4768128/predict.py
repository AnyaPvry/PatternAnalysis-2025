from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch

SAVED_PATH = "./saved_models/final" 

model = T5ForConditionalGeneration.from_pretrained(SAVED_PATH)
tokenizer = T5Tokenizer.from_pretrained(SAVED_PATH)

