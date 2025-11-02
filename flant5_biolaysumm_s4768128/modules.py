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
    # 5. Initialize Model
    tokenizer, model, data_collator = load_model()

    # 2. Load and preprocess dataset
    subset_train, subset_val = load_clean_dataset()
    tokenized_train, tokenized_val = preprocess_dataset(subset_train, subset_val, tokenizer)

    # 6. Fine-tuning
    trainer = fine_tune_model(model, tokenizer, data_collator, tokenized_train, tokenized_val)

    # visualize loss
    plot_training_curve(trainer)

    # Generate and display example summary
    sample = subset_val[0]
    print("\n--- Example Generation ---")
    print("Report:\n", sample["radiology_report"][:300], "...\n")
    print("Gold summary:\n", sample["layman_report"], "\n")
    print("Model summary:\n", generate_lay_summary(model, tokenizer, sample["radiology_report"]))

