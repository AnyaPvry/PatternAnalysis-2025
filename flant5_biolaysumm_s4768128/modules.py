from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq
from dataset import load_clean_dataset, preprocess_dataset


MODEL_NAME = "google/flan-t5-base"

def load_model():
    """Load FLAN-T5 model and tokenizer"""
    # 5. Initialize Model FLAN-T5-base
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    return tokenizer, model, data_collator

if __name__ == "__main__":
    tokenizer, model, data_collator = load_model()

    subset_train, subset_val = clean_dataset()
    tokenized_train, tokenized_val = preprocess_dataset(subset_train, subset_val, tokenizer)
    
    

