import os
import numpy as np
import nltk
import evaluate
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

# -- If saving model on local device --
# 0. make sure the folder exists
SAVE_DIR = "./saved_models/final"
os.makedirs(SAVE_DIR, exist_ok=True)

# -- If saving model on google drive, when working on google collab --
# from google.colab import drive
# drive.mount('/content/drive')  # authorize access

# Metric
metric = evaluate.load("rouge")

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # replace -100 with pad token so we can decode
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # sentence split for rougeLSum
    decoded_preds = ["\n".join(nltk.sent_tokenize(p.strip())) for p in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(l.strip())) for l in decoded_labels]

    result = metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
    )
    return result

# Training arguments
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

training_args = Seq2SeqTrainingArguments(
    output_dir="/content/drive/MyDrive/saved_models/",
    eval_strategy="epoch",             # use new argument name
    learning_rate=2e-4,             # smaller lr → more stable
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,   # 👈 add this
    weight_decay=0.01,
    num_train_epochs=3,
    fp16=False,                     # absolutely disable mixed precision
    bf16=False,                     # also disable bfloat16
    logging_steps=10,
    logging_first_step=True,
    predict_with_generate=True,
    report_to="none",
    remove_unused_columns=False,
    label_names=["labels"],         # tell trainer where to find labels
    #max_grad_norm=1.0,              # clip gradients to avoid explosions
)

# Force model to FP32 just in case
model = model.to("cuda").float()