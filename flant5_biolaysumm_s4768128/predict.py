# test.py
import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from evaluate import load

# Load the fine-tuned model 
def load_finetuned_model(saved_path="./saved_models/final"):
    """Load the fine-tuned model and tokenizer from disk"""
    model = T5ForConditionalGeneration.from_pretrained(saved_path)
    tokenizer = T5Tokenizer.from_pretrained(saved_path)

    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    return model, tokenizer


# Tokenize test data 
def tokenize_test_data(dataset, preprocess_function, cols_to_keep, sample_size=1000):
    """Prepare and tokenize a subset of the test dataset"""
    subset_test = dataset["test"].shuffle(seed=42).select(range(sample_size))

    tokenized_test = subset_test.map(
        preprocess_function,
        batched=True,
        remove_columns=[c for c in subset_test.column_names if c not in cols_to_keep],
    )
    return subset_test, tokenized_test


# Generate predictions 
def generate_predictions(model, tokenizer, tokenized_test, output_dir="/content/tmp-pred"):
    """Generate model predictions for the test set"""
    # Define generation arguments
    gen_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=16,
        predict_with_generate=True,
        report_to="none",
        generation_max_length=90,
        generation_num_beams=4,
    )

    # Create temporary trainer for prediction
    pred_trainer = Seq2SeqTrainer(
        model=model,
        args=gen_args,
        processing_class=tokenizer,
    )

    # Run model on test set
    pred_output = pred_trainer.predict(tokenized_test)

    # Extract generated token IDs and decode to text
    pred_ids = pred_output.predictions
    decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    return decoded_preds

# Display a few test examples
def display_examples(results, num_examples=5):
    """
    Display first few examples of model input, ground truth, and prediction.
    results: list of tuples -> (input_text, ground_truth, prediction)
    """
    print("\n=== Example Model Outputs ===")
    for i, (inp, gt, pred) in enumerate(results[:num_examples]):
        print(f"\n--- Example {i+1} ---")
        print("INPUT:", inp[:250], "...")
        print("GROUND TRUTH:", gt)
        print("MODEL OUTPUT:", pred)

# Compute ROUGE metrics 
def compute_rouge_from_results(results):
    """
    Compute ROUGE metrics using the evaluate library.
    results: list of tuples -> (input_text, ground_truth, prediction)
    """
    rouge = load("rouge")

    # Extract model outputs and references
    preds = [r[2] for r in results]  # generated summaries
    refs  = [r[1] for r in results]  # ground-truth summaries

    # Compute ROUGE scores
    scores = rouge.compute(predictions=preds, references=refs)
    print("\n=== ROUGE Evaluation Results ===")
    print(scores)

    return scores