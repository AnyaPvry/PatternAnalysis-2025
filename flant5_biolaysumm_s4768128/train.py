import os
import numpy as np
import nltk
import evaluate
import matplotlib.pyplot as plt
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

# Compute ROUGE metrics
def compute_metrics(eval_preds, tokenizer):
    """Compute ROUGE metrics for validation/evaluation"""
    preds, labels = eval_preds

    # Replace -100 (ignored positions) with pad_token_id for decoding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode token IDs back to text
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Split into sentences for ROUGE-LSum
    decoded_preds = ["\n".join(nltk.sent_tokenize(p.strip())) for p in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(l.strip())) for l in decoded_labels]

    # Compute ROUGE scores
    metric = evaluate.load("rouge")
    return metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

# 2. Fine-tuning and training loop
def fine_tune_model(model, tokenizer, data_collator, tokenized_train, tokenized_val, save_dir="./saved_models/final"):
    """Train and evaluate the FLAN-T5 model"""
    os.makedirs(save_dir, exist_ok=True)  # ensure save directory exists

    # Define training configuration
    training_args = Seq2SeqTrainingArguments(
        output_dir=save_dir,                 # where to save checkpoints
        evaluation_strategy="epoch",         # evaluate after each epoch
        learning_rate=2e-4,                  # tuned LR for stable training
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        weight_decay=0.01,
        num_train_epochs=3,
        logging_steps=10,
        logging_first_step=True,
        predict_with_generate=True,          # generate text during eval
        push_to_hub=False,
        report_to="none",                    # no external logging
    )

    # Move model to GPU and force full precision
    model = model.to("cuda").float()

    # Initialize Hugging Face Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, tokenizer),  # attach metrics
    )

    # Start fine-tuning
    trainer.train()

    # Save model and tokenizer AFTER training completes
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Final model and tokenizer saved to: {save_dir}")

    return trainer

# Plot training loss curve
def plot_training_curve(trainer):
    """Plot training loss to visualize convergence"""
    logs = trainer.state.log_history
    train_steps = [entry["step"] for entry in logs if "loss" in entry]
    train_loss = [entry["loss"] for entry in logs if "loss" in entry]

    plt.figure(figsize=(7, 4))
    plt.plot(train_steps, train_loss, label="Training Loss")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("FLAN-T5 Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.show()
    

# Generate lay summary for a single report
def generate_lay_summary(model, tokenizer, radiology_report):
    """Generate lay summary from a single radiology report"""
    PREFIX = "Summarize this radiology report for a layperson: "
    text = PREFIX + radiology_report
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    GR = model.generate(**inputs, num_beams=2, max_new_tokens=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

