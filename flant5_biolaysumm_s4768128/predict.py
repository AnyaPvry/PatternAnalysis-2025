from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch

SAVED_PATH = "./saved_models/final" 

model = T5ForConditionalGeneration.from_pretrained(SAVED_PATH)
tokenizer = T5Tokenizer.from_pretrained(SAVED_PATH)

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

print("Model and tokenizer loaded successfully")

# Preprocessing
subset_test = dataset['test'].shuffle(seed=42).select(range(1000))
tokenized_test = subset_test.map(
    preprocess_function,
    batched=True,
    remove_columns=[c for c in subset_test.column_names if c not in cols_to_keep],
)

# dummy args for trainer
gen_args = Seq2SeqTrainingArguments(
    output_dir="/content/tmp-pred", # temp dir
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    report_to="none",
    generation_max_length=90,
    generation_num_beams=4, 
)

pred_trainer = Seq2SeqTrainer(
    model=model,
    args=gen_args,
    processing_class=tokenizer,
)