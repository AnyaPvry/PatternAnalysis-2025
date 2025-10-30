from transformers import T5Tokenizer, T5ForConditionalGeneration
from dataset import load_clean_dataset, preprocess_dataset

MODEL_NAME = "google/flan-t5-base"

def load_model():
    """Load FLAN-T5 model and tokenizer"""
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    return tokenizer, model

if __name__ == "__main__":
    tokenizer, model = load_model()

    dataset = load_clean_dataset()


