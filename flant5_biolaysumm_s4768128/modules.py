# modules.py
"""
modules.py
Defines and loads the FLAN-T5 model and tokenizer for the BioLaySumm task.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# model name from Hugging Face Hub
MODEL_NAME = "google/flan-t5-small"   # you can upgrade to flan-t5-base later

def load_model():
    """Load the pretrained FLAN-T5 model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return tokenizer, model


if __name__ == "__main__":
    # quick sanity check
    tokenizer, model = load_model()
    print("Loaded:", MODEL_NAME)
    inputs = tokenizer("Translate the radiology report into lay terms: Bilateral peribronchovascular thickening.", return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
