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
## Folder Structure
1. modules.py: loads the flan-t5 model, tokenizer, seq2seq data collator, and is where main is located. Does not follow task sheet suggestion of having source code for compoenets of my model since Flan-T5 is a pre-trained model from Hugging Face, its architecture is already implemented and only needs to be loaded, not redefined.

2. dataset.py: contians the data loader, preparation, and preprocessing functionality.

3. train.py: contains the training, validating, testing (tested traning with a subset of validation data) and saving of the fine-tuned model. Training results displayed includes a training loss curve, validation metrics, and example generation of one sample run.

4. predict.py: Tests the final fine tuned model on the whole validation dataset, with final outputs displaying example results and rouge scores.

- BioLaySumm_FlanT5_Finetuning.ipynb: code used to run on google colab.


## Implementation process run through

### 1. Data Loading and Preparation (dataset.py)
The BioLaySumm dataset is imported from hugging face library (BioLaySumm Shared Task at ACL, 2025).
The raw dataset comes in this structure:
- train 150k rows
- validation 10k rows
- test 10.5k rows (not used)
Each with the fields: source, images_path, radiology_report, layman_report.

Once the dataset is loaded, preparation is done to extract only specific data required for the purpose of the task.
This includes ommiting the test dataset because it does not come with any layman_report data, thus future evalidation with ground truth comparison would not be possible. 

As such, a custom clean_up function is applied to only the train and validation datasets, where further extraction of only the radiology_report and layman_report fields processed and used. The processing entails striping of leading and trailing spaces or newline characters from each entry, and removing any examples where either the radiology report or layman report was missing or empty. Hence future tokenization does not have to be perfomed on empty strings or fields, wasting learning capacity.

A subset of each datast is also chosen through selecting a fixed range; this is set for prototyping on a smaller instruction dataset, then scaling and readjustment with trial and error to find optimal dataset size. 
Shuffling with fixed seed is also set for reproducability and ensures the training data is randomized in a consistent manner across runs, preventing any bias from the original dataset order while maintaining deterministic reproducibility.

### 2. Data Formatting and Preprocessing (modules.py)
This stage prepares the dataset for input into the Flan-T5 model. The tokenizer is first loaded in modules.py and is responsible for converting raw text into model-readable numerical format. Specifically, the tokenizer splits each text sequence into subword tokens and maps these tokens into token IDs, which correspond to entries in the model’s vocabulary.

Each radiology report is prefixed with an instruction prompt for instruction fine tuning—
"Summarize this radiology report for a layperson: " —

The function preprocess_dataset() applies tokenization separately to the training and validation subsets. For each sample, two main text fields are processed:

- Input (radiology_report): Tokenized with a maximum input length of 256 tokens, reflecting the typically longer and more detailed nature of radiology reports.
- Target (layman_report): Tokenized with a shorter maximum length of 128 tokens, as lay summaries are expected to be more concise.

Padding is set to "longest", meaning all sequences in a batch are padded to the length of the longest example to ensure equal tensor dimensions for efficient batch processing. Truncation ensures any text longer than the maximum length is cut off rather than exceeding the model’s context window.

After tokenization, three key components are returned for each example in the dataset:

```
{
  "input_ids": tensor([...]),        # Token IDs representing the input (radiology report)
  "attention_mask": tensor([...]),   # Binary mask indicating which tokens are actual text (1) vs. padding (0)
  "labels": tensor([...])            # Token IDs for the target summary, used as decoder labels during training
}
```

The attention_mask guides the model to focus only on valid tokens during computation, ignoring padded positions. The labels correspond to the ground-truth lay summaries and are used by the loss function to compare predicted tokens with expected ones.

Finally, the processed datasets are returned as tokenized_train and tokenized_val, each containing only the columns needed for training — input_ids, attention_mask, and labels. This ensures a clean, structured format ready for the model’s fine-tuning pipeline.

(“Tokenizer,” 2018)

### 3. Collate and Batching (modules.py)
Instruction fine tuning data preparation:
After tokenization, the dataset still consists of Python lists of token IDs with variable lengths. However, the model expects each batch to be a set of tensors with the same sequence length. To handle this automatically, we use the Hugging Face utility:
```
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
```
This automatically pads all inputs and targets in each batch to the same length, generates attention masks so the model ignores padding, and replaces padded label positions with -100 so they do not affect the loss.

It also prepares the decoder inputs by shifting the target tokens one position to the right, enabling the model to learn to predict the next token in the sequence. This step is essential for sequence-to-sequence fine-tuning, allowing the decoder to generate fluent, coherent summaries token by token.

### 4. Create Data Loader
In this project, the Seq2SeqTrainer automatically creates and manages the data loaders internally using the preprocessed datasets and DataCollatorForSeq2Seq. Therefore, no separate DataLoader initialization is required.

### 5. Initialize Model FLAN-T5-base (modules.py)

### 6. Fine-tuning ()
...

### 7. Generate and Save Responses


### 8. Model Evaluation

Tested on all 10k validation dataset

---
## Results

**Trials**
| Trial no. | Test | Validation | learning rate |
| --- | --- | --- | -- |
| Trial 0 | 2000 | 200 | 3e-4
| Trial 1 | 2000 | 200 | 2e-4
| Trial 2 | 20,000 | 2000 | 3e-4
| Trial 3 | 20,000 | 2000 | 2e-4

### Trial 0: Test 2000, Validation 200, Learning Rate 3e-4
**In Training**
Table

loss curve

Validation metrics: {'validation_loss': 0.5616681575775146, 'validation_rouge1': 0.4005814744200473, 'validation_rouge2': 0.22337463515126288, 'validation_rougeL': 0.34938132175511655, 'validation_rougeLsum': 0.3699470848351884, 'validation_runtime': 7.187, 'validation_samples_per_second': 27.828, 'validation_steps_per_second': 0.974, 'epoch': 3.0}

--- Example Generation ---
Report:
 The chest shows significant air trapping. Bilateral apical chronic changes are present. Dorsal kyphosis is noted. No evidence of pneumothorax. ...

Ground Truth summary:
 The chest shows a large amount of trapped air. There are long-term changes at the top of both lungs. The upper back is curved outward. There is no sign of air in the space around the lungs. 

Model summary:
 The chest shows significant air trapping. There are chronic changes in the apical apical changes. There is also a kyphosis in the lungs. There is no sign of pneumothorax.

### Trial 1: Test 2000, Validation 200, Learning Rate 2e-4
**In Training:**
table

loss curve

Validation metrics: {'validation_loss': 0.4926953911781311, 'validation_rouge1': 0.43138340708215817, 'validation_rouge2': 0.2595719856552169, 'validation_rougeL': 0.3793345296400347, 'validation_rougeLsum': 0.40013170566378886, 'validation_runtime': 7.9915, 'validation_samples_per_second': 25.026, 'validation_steps_per_second': 0.876, 'epoch': 3.0}

--- Example Generation ---
Report:
 The chest shows significant air trapping. Bilateral apical chronic changes are present. Dorsal kyphosis is noted. No evidence of pneumothorax. ...

Ground Truth summary:
 The chest shows a large amount of trapped air. There are long-term changes at the top of both lungs. The upper back is curved outward. There is no sign of air in the space around the lungs. 

Model summary:
 The chest shows significant air trapping. There are long-term changes in both apical apical areas. There is also a curvature of the spine. There is no sign of pneumonia in the lungs.

**In Testing:**
--- Example 1 ---
INPUT: The chest shows significant air trapping. Bilateral apical chronic changes are present. Dorsal kyphosis is noted. No evidence of pneumothorax. ...
GROUND TRUTH: The chest shows a large amount of trapped air. There are long-term changes at the top of both lungs. The upper back is curved outward. There is no sign of air in the space around the lungs.
MODEL OUTPUT: The chest x-ray shows a lot of air is trapped in the lungs. There are long-term changes at the top of both lungs. The upper back is curved outward. There is no sign of air leakage outside the lungs.

--- Example 2 ---
INPUT: Central venous catheter traversing the left jugular vein with its tip in the superior vena cava. The remainder is unchanged. ...
GROUND TRUTH: A central venous catheter is going through the left jugular vein and its tip is in the superior vena cava. Everything else is the same as before.
MODEL OUTPUT: A central venous catheter is going through the left jugular vein and its tip is in the superior vena cava. Everything else looks the same as before.

--- Example 3 ---
INPUT: Chronic pulmonary changes ...
GROUND TRUTH: Long-term changes in the lungs are seen.
MODEL OUTPUT: Long-term changes in the lungs are seen.

--- Example 4 ---
INPUT: Radiological signs of air trapping, flattened diaphragm, and increased retrosternal space. Calcified pleural plaques at the level of the left diaphragmatic pleura. Loss of volume in the left lung with subpleural linear opacities. Findings are related ...
GROUND TRUTH: The X-ray shows signs of trapped air, a flattened muscle under the lungs, and more space behind the breastbone. There are also hardened areas on the lung lining on the left side. The left lung has lost some volume and has some linear shadows near the outer lining. These findings are related to long-term inflammation caused by exposure to asbestos. Looking at the previous CT scan, there are no significant changes compared to the scanogram dated 3/4/2009.
MODEL OUTPUT: The x-ray shows signs of air being trapped in the lungs, flattened diaphragm, and increased space behind the breastbone. There are calcified plaques on the left side of the diaphragm. The left lung has less volume with linear opacities. These findings are related to long-term inflammation caused by exposure to asbestos. The previous CT scan shows no significant changes compared to the one taken on 3/4/2009.

--- Example 5 ---
INPUT: Calcified granuloma in the right lung vertex. ...
GROUND TRUTH: There is a calcified granuloma located at the top of the right lung.
MODEL OUTPUT: There is a calcified granuloma, which is a type of hardened lump, in the right lung area.

Final ROUGE Scores:
{'rouge1': np.float64(0.7050470905154624), 'rouge2': np.float64(0.5288976000819363), 'rougeL': np.float64(0.6579951511060707), 'rougeLsum': np.float64(0.6580987039946229)}

### Trial 2: Test 200,000, Validation 2000, Learning Rate 3e-4
**In Training:**
Table

loss curve

Validation metrics: {'validation_loss': 0.30140259861946106, 'validation_rouge1': 0.5303797909718861, 'validation_rouge2': 0.38963944906846537, 'validation_rougeL': 0.49262804248382636, 'validation_rougeLsum': 0.5064069921726673, 'validation_runtime': 71.4698, 'validation_samples_per_second': 27.984, 'validation_steps_per_second': 0.881, 'epoch': 3.0}

--- Example Generation ---
Report:
 The chest shows significant air trapping. Bilateral apical chronic changes are present. Dorsal kyphosis is noted. No evidence of pneumothorax. ...

Ground Truth summary:
 The chest shows a large amount of trapped air. There are long-term changes at the top of both lungs. The upper back is curved outward. There is no sign of air in the space around the lungs. 

Model summary:
 The chest shows significant air trapping. There are long-term changes at the top of both lungs. There is a curvature of the spine in the upper back. There is no sign of air in the chest cavity.

**In Testing:**
--- Example 1 ---
INPUT: The chest shows significant air trapping. Bilateral apical chronic changes are present. Dorsal kyphosis is noted. No evidence of pneumothorax. ...
GROUND TRUTH: The chest shows a large amount of trapped air. There are long-term changes at the top of both lungs. The upper back is curved outward. There is no sign of air in the space around the lungs.
MODEL OUTPUT: The chest shows significant air trapping. There are long-term changes at the top of both lungs. There is a curvature of the spine in the upper back. There is no sign of air in the chest cavity.

--- Example 2 ---
INPUT: Central venous catheter traversing the left jugular vein with its tip in the superior vena cava. The remainder is unchanged. ...
GROUND TRUTH: A central venous catheter is going through the left jugular vein and its tip is in the superior vena cava. Everything else is the same as before.
MODEL OUTPUT: A central venous catheter is going through the left jugular vein and its tip is in the superior vena cava. Everything else looks the same.

--- Example 3 ---
INPUT: Chronic pulmonary changes ...
GROUND TRUTH: Long-term changes in the lungs are seen.
MODEL OUTPUT: Long-term changes in the lungs are seen.

--- Example 4 ---
INPUT: Radiological signs of air trapping, flattened diaphragm, and increased retrosternal space. Calcified pleural plaques at the level of the left diaphragmatic pleura. Loss of volume in the left lung with subpleural linear opacities. Findings are related ...
GROUND TRUTH: The X-ray shows signs of trapped air, a flattened muscle under the lungs, and more space behind the breastbone. There are also hardened areas on the lung lining on the left side. The left lung has lost some volume and has some linear shadows near the outer lining. These findings are related to long-term inflammation caused by exposure to asbestos. Looking at the previous CT scan, there are no significant changes compared to the scanogram dated 3/4/2009.
MODEL OUTPUT: The x-ray shows signs of air being trapped in the lungs, flattened diaphragm, and increased space behind the breastbone. There are calcified plaques at the level of the left diaphragm pleura. The left lung has less volume and hazy areas near the lung surface. These findings are related to long-term inflammation caused by exposure to asbestos. Compared to

--- Example 5 ---
INPUT: Calcified granuloma in the right lung vertex. ...
GROUND TRUTH: There is a calcified granuloma located at the top of the right lung.
MODEL OUTPUT: There is a calcified granuloma, which is a type of hardened lump, in the right lung area.

Final ROUGE Score:
- {'rouge1': np.float64(0.6884713874885872), 'rouge2': np.float64(0.5090535549465793), 'rougeL': np.float64(0.6399409171921827), 'rougeLsum': np.float64(0.640003186291338)}

### Trial 3: Test 20,000, Validation 2000, Learning Rate 2e-4
**In Training:**
table

graph

Validation metrics: {'validation_loss': 0.32267871499061584, 'validation_rouge1': 0.5189141267183089, 'validation_rouge2': 0.37682674160952473, 'validation_rougeL': 0.48067478947913755, 'validation_rougeLsum': 0.49497094889873633, 'validation_runtime': 65.8633, 'validation_samples_per_second': 30.366, 'validation_steps_per_second': 0.957, 'epoch': 3.0}

--- Example Generation ---
Report:
 The chest shows significant air trapping. Bilateral apical chronic changes are present. Dorsal kyphosis is noted. No evidence of pneumothorax. ...

Ground Truth summary:
 The chest shows a large amount of trapped air. There are long-term changes at the top of both lungs. The upper back is curved outward. There is no sign of air in the space around the lungs. 

Model summary:
 The chest shows significant air trapping. There are long-term changes in both lower parts of the lungs. There is a curvature of the spine in the upper back. There is no sign of air in the chest cavity.

**In Testing**
--- Example 1 ---
INPUT: The chest shows significant air trapping. Bilateral apical chronic changes are present. Dorsal kyphosis is noted. No evidence of pneumothorax. ...
GROUND TRUTH: The chest shows a large amount of trapped air. There are long-term changes at the top of both lungs. The upper back is curved outward. There is no sign of air in the space around the lungs.
MODEL OUTPUT: The chest shows significant air trapping. There are long-term changes in both lower parts of the lungs. There is a curvature of the spine in the upper back. There is no sign of air in the chest cavity.

--- Example 2 ---
INPUT: Central venous catheter traversing the left jugular vein with its tip in the superior vena cava. The remainder is unchanged. ...
GROUND TRUTH: A central venous catheter is going through the left jugular vein and its tip is in the superior vena cava. Everything else is the same as before.
MODEL OUTPUT: A central venous catheter is going through the left jugular vein and its tip is in the superior vena cava. Everything else looks the same.

--- Example 3 ---
INPUT: Chronic pulmonary changes ...
GROUND TRUTH: Long-term changes in the lungs are seen.
MODEL OUTPUT: Long-term changes in the lungs are seen.

--- Example 4 ---
INPUT: Radiological signs of air trapping, flattened diaphragm, and increased retrosternal space. Calcified pleural plaques at the level of the left diaphragmatic pleura. Loss of volume in the left lung with subpleural linear opacities. Findings are related ...
GROUND TRUTH: The X-ray shows signs of trapped air, a flattened muscle under the lungs, and more space behind the breastbone. There are also hardened areas on the lung lining on the left side. The left lung has lost some volume and has some linear shadows near the outer lining. These findings are related to long-term inflammation caused by exposure to asbestos. Looking at the previous CT scan, there are no significant changes compared to the scanogram dated 3/4/2009.
MODEL OUTPUT: The x-ray shows signs of air being trapped in the lungs, flattened diaphragm, and increased space behind the breastbone. There are calcified plaques at the level of the left diaphragm pleura. The left lung has less volume with hazy areas below the pleura. These findings are related to long-term inflammation due to asbestos exposure. Compared to the scan from 3/4/2009, there are no significant changes.

--- Example 5 ---
INPUT: Calcified granuloma in the right lung vertex. ...
GROUND TRUTH: There is a calcified granuloma located at the top of the right lung.
MODEL OUTPUT: A calcified granuloma is present in the right lung area.

Final ROUGE Score:
- {'rouge1': np.float64(0.6839410460453273), 'rouge2': np.float64(0.4984734093640375), 'rougeL': np.float64(0.6336842837157162), 'rougeLsum': np.float64(0.6335927961552938)}

---
## Conclusion

###Trial 1 — (Train: 2000 | Val: 200 | LR = 3e-4)
Test trial on small dataset and no test data was recorded

**Training Results**
Table & Graph: The loss curve is smooth, showing a steep early drop followed by a consistent flattening trend, meaning the model quickly minimized loss, a faster more aggressive learning, and then stabilized.

Validation/ROUGE Metrics:
- validation_loss: 0.5616
- ROUGE-1: 0.40, ROUGE-2: 0.22, ROUGE-L: 0.35

Overall performance was modest, showing that the model captured basic patterns but lacked vocabulary diversity due to small data volume.

Examples: Model outputs closely resembled the input phrasing with minor structural errors (e.g., repetition such as “apical apical changes”). It correctly captured the meaning but with poor lexical variety and redundancy.

###Trial 2 — (Train: 2000 | Val: 200 | LR = 2e-4)
Test trial on same small dataset but with lower learning rate

**Testing Results**
Table & Graph: The loss curve has several local peaks and dips; it declines more gradually and oscillates mildly - less smooth than trial 1.

Validation/ROUGE Metrics:
- validation_loss: 0.4927
- ROUGE-1: 0.43, ROUGE-2: 0.26, ROUGE-L: 0.38

Clear improvement in all metrics, suggesting better generalization and sentence structure accuracy.

Examples: Generated training examples were more fluent and semantically faithful to the ground truth with improved phrasing and reduced redundancy.

**Testing Results**

Examples: The model performed well across unseen samples, producing clear and human-readable summaries closely matching ground truths. Examples showed accurate translations of medical phrasing into lay terms.

Final ROUGE Score:
- ROUGE-1: 0.7050, ROUGE-2: 0.5289, ROUGE-L: 0.6580

These higher scores indicate stronger alignment with reference summaries and effective fine-tuning even on a small dataset.

Trial 3 — (Train: 20,000 | Val: 2,000 | LR = 3e-4)
Scale up dataset by ten fold.

**Training Results**

Table & Graph: Loss dropped steadily with a consistent downward trend, showing efficient learning on the larger dataset. Training was smooth with no instability observed.

Validation/ROUGE Metrics:
- validation_loss: 0.3014
- ROUGE-1: 0.53, ROUGE-2: 0.39, ROUGE-L: 0.49

Substantial improvement across all metrics, demonstrating that scaling data significantly enhanced summarization accuracy and diversity.

Examples: Outputs were coherent and faithful to input meaning, accurately reformulating medical terminology into understandable layman language.

**Testing Results**
Examples: Test results showed precise, concise, and medically correct paraphrasing. The summaries preserved context and reduced repetition.

Final ROUGE Score:
- ROUGE-1: 0.6885, ROUGE-2: 0.5090, ROUGE-L: 0.6399

Strong results indicating balanced generalization and fluency, with near-human readability in summaries.

Trial 4 — (Train: 20,000 | Val: 2,000 | LR = 2e-4)

**Training Results**
Table & Graph: The model converged smoothly with a slightly slower rate than Trial 3, suggesting more conservative learning. Training remained stable across all epochs.

Validation/ROUGE Metrics:
- validation_loss: 0.3227
- ROUGE-1: 0.52, ROUGE-2: 0.38, ROUGE-L: 0.48

Comparable to Trial 3 with consistent quality and reliability, but slightly lower.

Examples: Generated samples were fluent and medically accurate, reflecting correct understanding of radiology context with minimal grammatical errors.

**Testing Results**

Examples: Test outputs were concise and contextually aligned with ground truths, maintaining accurate terminology and natural phrasing.

Final ROUGE Score:
- ROUGE-1: 0.6839, ROUGE-2: 0.4985, ROUGE-L: 0.6336

Slightly below Trial 3 but still demonstrating robust summarization quality with stable language generation.

---
## Evalution

Final ROUGE scores
| Trial no. | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | --- | --- | -- |
| Trial 1 | 0.7050 | 0.5289 | 0.6580
| Trial 2 | 0.6885 | 0.5090 | 0.6399
| Trial 3 | 0.6839 | 0.4985 | 0.6336

Overall, Trial 3 achieved the best balance of accuracy and stability. Increasing the dataset size from 2 k to 20 k examples led to clear gains in ROUGE scores and lower validation loss, showing that data scale was the primary driver of performance improvement - larger data volumes allow the model to learn richer vocabulary and generalize better to unseen samples. The learning-rate experiments showed that, in the small-data runs, the lower rate (2 × 10⁻⁴) caused more oscillation and slower convergence, while the higher rate (3 × 10⁻⁴) produced a smoother loss curve and faster stabilization. On the larger dataset, both rates trained stably, but 3 × 10⁻⁴ reached the best overall metrics. Hence, performance correlated most strongly with dataset size, and the optimal setup combined a large dataset with the slightly higher learning rate (Trial 3).

If more time was allowed, for future projects I would do more testing on even larger datasets, and try to experiment with adjusting other training parameters as well. 


Building an LLM
 - 

 Tokenization

 After pretraining, the model knows general language, but not how to follow human instructions. Instruction fine-tuning ttranins it on prias of instructions and desired replies so it learnss to respond in the way people expect

---