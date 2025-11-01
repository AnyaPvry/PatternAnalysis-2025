# Pattern Analysis
Pattern Analysis of various datasets by COMP3710 students in 2025 at the University of Queensland.

We create pattern recognition and image processing library for Tensorflow (TF), PyTorch or JAX.

This library is created and maintained by The University of Queensland [COMP3710](https://my.uq.edu.au/programs-courses/course.html?course_code=comp3710) students.

The library includes the following implemented in Tensorflow:
* fractals 
* recognition problems

In the recognition folder, you will find many recognition problems solved including:
* segmentation
* classification
* graph neural networks
* StyleGAN
* Stable diffusion
* transformers
etc.

# Fine Tuning FLAN-T5 on BioLaySumm dataset.

## Introduction
This task performs fine tuning on a pretrained model, FLAN-T5, to translate expert radiology reports into layperson summaries using the BioLaySumm dataset. The goal is to build a summarization model that produces clear, human-readable reports suitable for patients and non-medical readers. 

## Problem Space
Medical imaging reports are written by radiologists for specialists, often containing complex terminology and syntax. This makes them difficult for patients to understand.  
Using a sequence-to-sequence large language model (FLAN-T5), we fine-tune on paired radiology and layperson summaries to generate simplified, accessible versions.


## Dependencies
To reproduce training environment, install the following packages.

pip install -q --no-cache-dir \
  "transformers==4.46.3" \
  "tokenizers==0.20.1" \
  "accelerate==0.34.2" \
  "peft==0.13.2" \
  sentencepiece safetensors

pip install datasets evaluate rouge-score torch tensorboard

---
## Full code process run through

### 1. Data Loading and Preparation (dataset.py)
The BioLaySumm dataset is imported from hugging face library (BioLaySumm Shared Task at ACL, 2025).
The raw dataset comes in this structure:
- train 150k rows
- validation 10k rows
- test 10.5k rows
Each with the fields: source, images_path, radiology_report, layman_report.

Once the dataset is loaded, preparation is done to extract only specific data required for the purpose of the task.
This includes ommiting the test dataset because it does not come with any layman_report data, thus future evaluation with ground truth comparison would not be possible. 

As such, a custom clean_up function is applied to only the train and validation datasets, where further extraction of only the radiology_report and layman_report fields processed and used. The processing entails striping of leading and trailing spaces or newline characters from each entry, and removing any examples where either the radiology report or layman report was missing or empty. Hence future tokenization does not have to be perfomed on empty strings or fields, wasting learning capacity.

A subset of each datast is also chosen through selecting a fixed range; this is set for prototyping on a smaller instruction dataset, then scaling and readjustment with trial and error to find optimal dataset size. 
Shuffling with fixed seed is also set for reproducability and ensures the training data is randomized in a consistent manner across runs, preventing any bias from the original dataset order while maintaining deterministic reproducibility.

### 2. Data Formatting and Preprocessing (dataset.py)

The prepared data is passed through a tokenizer, with 
the context window for maximum input length set at 256 since radiology reports are longer, and maximmum target length at a shorter 128 since layman summaries are shorter.
Padding is set to 
```
{
  "input_ids": tensor([...]),
  "attention_mask": tensor([...]),
  "labels": tensor([...])
}
```

### 3. Collate and Batching (dataset.py)
For encoder-decoder models like T5 or FLAN-T5, `DataCollatorForSeq2Seq` does a few special things automatically.
```
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
```
- Dynamic padding per batch: Instead of padding everything to a fixed maximum length (wastes memory), it pads only up to the longest sequence in the batch.
 It pads both encoder inputs (radiology reports) and decoder targets (layperson summaries).
- Shifts the decoder labels (important for seq2seq): In an encoder-decoder model, the decoder predicts the next token in the target sequence. To make that work correctly, the input to the decoder is the target sequence shifted right by one token. The label is the original sequence (unshifted). This ensures the model only sees previous tokens, not the current or future ones.
This “shift right” behavior is like a teacher forcing setup, preventing the model from cheating.
- It creates the proper attention masks

It automatically generates: attention_mask for encoder inputs (which tokens are real vs. padding)
decoder_attention_mask for targets
These masks tell the model which tokens to attend to during forward passes.
This all gets packaged into a single dictionary that can be passed directly into the model during training.


Without a data collator, this would have to be manually done through manual padding of sequences for each branch, shifting the decoder labels, and writing the logic for attention mask. , So instead of you doing that manually, the collator handles it automatically.

### 4. Create Data Loader

### 5. Initialize Model FLAN-T5-base (dataset.py)
...

### 6. Fine-tuning ()
...

### 7. Generate and Save Responses

### 8. Model Evaluation

This code runs after training to perform a quick qualitative evaluation on a validation sample. It does not update model weights, does not compute gradients, and does not affect training in any way. It simply takes one radiology report from the validation set, shows the true lay summary, and prints the summary generated by the trained model so you can visually inspect its performance.


---

## Results
| Trial no. | Test | Validation | learning rate |
| --- | --- | --- | -- |
| Trial 1 | 2000 | 200 | 3e-4
| Trial 2 | 2000 | 200 | 2e-4
| Trial 2 | 20,000 | 2000 | 3e-4


### Trial 2: Test 200,000, Validation 2000

Table

loss curve


Valuation metrics: {'valuation_loss': 0.30140259861946106, 'valuation_rouge1': 0.5303797909718861, 'valuation_rouge2': 0.38963944906846537, 'valuation_rougeL': 0.49262804248382636, 'valuation_rougeLsum': 0.5064069921726673, 'valuation_runtime': 71.4698, 'valuation_samples_per_second': 27.984, 'valuation_steps_per_second': 0.881, 'epoch': 3.0}


--- Example Generation ---
Report:
 The chest shows significant air trapping. Bilateral apical chronic changes are present. Dorsal kyphosis is noted. No evidence of pneumothorax. ...

Ground Truth summary:
 The chest shows a large amount of trapped air. There are long-term changes at the top of both lungs. The upper back is curved outward. There is no sign of air in the space around the lungs. 

Model summary:
 The chest shows significant air trapping. There are long-term changes at the top of both lungs. There is a curvature of the spine in the upper back. There is no sign of air in the chest cavity.

--- Example 0 ---
INPUT: The chest shows significant air trapping. Bilateral apical chronic changes are present. Dorsal kyphosis is noted. No evidence of pneumothorax. ...
GROUND TRUTH: The chest shows a large amount of trapped air. There are long-term changes at the top of both lungs. The upper back is curved outward. There is no sign of air in the space around the lungs.
MODEL OUTPUT: The chest shows significant air trapping. There are long-term changes at the top of both lungs. There is a curvature of the spine in the upper back. There is no sign of air in the chest cavity.

--- Example 1 ---
INPUT: Central venous catheter traversing the left jugular vein with its tip in the superior vena cava. The remainder is unchanged. ...
GROUND TRUTH: A central venous catheter is going through the left jugular vein and its tip is in the superior vena cava. Everything else is the same as before.
MODEL OUTPUT: A central venous catheter is going through the left jugular vein and its tip is in the superior vena cava. Everything else looks the same.

--- Example 2 ---
INPUT: Chronic pulmonary changes ...
GROUND TRUTH: Long-term changes in the lungs are seen.
MODEL OUTPUT: Long-term changes in the lungs are seen.

--- Example 3 ---
INPUT: Radiological signs of air trapping, flattened diaphragm, and increased retrosternal space. Calcified pleural plaques at the level of the left diaphragmatic pleura. Loss of volume in the left lung with subpleural linear opacities. Findings are related ...
GROUND TRUTH: The X-ray shows signs of trapped air, a flattened muscle under the lungs, and more space behind the breastbone. There are also hardened areas on the lung lining on the left side. The left lung has lost some volume and has some linear shadows near the outer lining. These findings are related to long-term inflammation caused by exposure to asbestos. Looking at the previous CT scan, there are no significant changes compared to the scanogram dated 3/4/2009.
MODEL OUTPUT: The x-ray shows signs of air being trapped in the lungs, flattened diaphragm, and increased space behind the breastbone. There are calcified plaques at the level of the left diaphragm pleura. The left lung has less volume and hazy areas near the lung surface. These findings are related to long-term inflammation caused by exposure to asbestos. Compared to

--- Example 4 ---
INPUT: Calcified granuloma in the right lung vertex. ...
GROUND TRUTH: There is a calcified granuloma located at the top of the right lung.
MODEL OUTPUT: There is a calcified granuloma, which is a type of hardened lump, in the right lung area.

{'rouge1': np.float64(0.6884713874885872), 'rouge2': np.float64(0.5090535549465793), 'rougeL': np.float64(0.6399409171921827), 'rougeLsum': np.float64(0.640003186291338)}

## Evalution

# Conclusion


Building an LLM
 - 

 Tokenization
 Split raw text into tokens and map these tokens into token IDs

 After pretraining, the model knows general language, but not how to follow human instructions. Instruction fine-tuning ttranins it on prias of instructions and desired replies so it learnss to respond in the way people expect

---