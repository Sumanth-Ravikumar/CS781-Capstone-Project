# Phase 2: Legal Named Entity Recognition (L-NER)

The task is to identify every entity in a long Indian court judgment and to output its character span and type. A prediction is counted as correct only if the start, the end and the label all match the gold annotation exactly (strict matching). Five adaptation techniques were applied to the four Phase 1 models.

## 1. Data and metric

- **Entity types (12).** APP (appellant), RESP (respondent), A.COUNSEL and R.COUNSEL (counsel for each side), JUDGE, WIT (witness), AUTH (authority), COURT, STAT (statute), PREC (precedent), DATE and CASENO.
- **Data.** 70 annotated judgments for training and 35 unlabelled judgments for the test leaderboard. All training documents exceed 1,024 tokens (the longest is about 23,700 tokens; the training set totals about 400,000 tokens). About 90.8% of all tokens belong to no entity, and there are no overlapping gold spans.
- **Validation split.** No development set is provided. The 70 documents were therefore split by document into 60 for training and 10 for validation (fixed seed 42). Validation scores are noisy: epoch-to-epoch and seed-to-seed variation of about 0.02 macro-F1 was measured. `split_and_oracle.json` records the split.
- **Output.** For each document, a list of `[start, end, label, entity_text]` entries, with `end` exclusive.
- **Metric.** Strict macro-F1 over the 12 types (secondary: macro-precision and macro-recall), computed by a scorer written for this project. A lenient score (any overlap with the correct label) is used only as a diagnostic.

## 2. Methodology

Named entity recognition is modelled as token classification on top of each Phase 1 backbone.

1. **Tokenisation with offsets.** Each document is encoded with the Phase 1 tokeniser while the character offset of every token is retained.
2. **BIO tagging.** Gold spans are projected onto tokens as B-/I-/O tags, which yields 25 classes (12 labels, two tags each, plus O).
3. **Windows.** Each document is cut into windows of 1,024 tokens with a stride of 768; the final window is anchored at the end of the document.
4. **Model.** The backbone reads each window; dropout (0.1) and a linear head classify every token. Decoder backbones are causal. The encoder-decoder contributes only its bidirectional encoder.
5. **Loss.** Cross-entropy with class weights equal to the square root of the inverse class frequency, clipped at 10, to counter the dominant background class.
6. **Optimisation.** AdamW with separate learning rates (backbone 1e-4 and head 1e-3 for full fine-tuning), 6% warm-up followed by linear decay, weight decay 0.01, batch size 8, bf16, and early stopping on validation strict macro-F1.
7. **Decoding.** Token probabilities from overlapping windows are averaged; the arg-max tags are grouped into spans, and leading and trailing whitespace is trimmed.
8. **Pipeline check.** Encoding the gold spans into tags and decoding them again reproduces the gold annotation with a strict macro-F1 of 0.9866, which is the ceiling of the tagging-and-decoding scheme.

## 3. Adaptation techniques

| Technique | Trained parameters | Description |
| --- | --- | --- |
| Full fine-tuning | All backbone weights and the head (87.5M for the GQA model) | Separate backbone and head learning rates |
| Knowledge distillation | Half-size student built from random weights (43.2M, 49% of the teacher) | Loss = (1 - a) CE + a T^2 KL(teacher, student), with a = 0.5 and T = 2 |
| LoRA | Low-rank matrices A and B on the q, k, v and output projections of every layer, plus the head | y = W0 x + (a/r) B(Ax); B is initialised at zero; r = 8, a = 16 |
| LoRMA+ | Low-rank matrices on the output side of the same projections, plus the head | Multiplicative update ((a/r) B A + I) W0; identity at initialisation and mergeable into the weights; r = 10, a = 20 (r = 9 for the encoder-decoder) |
| Prefix tuning | 32 trainable key and value vectors per layer, plus the head | Prefixes are passed through the attention-cache interface and initialised from the keys and values of 32 real tokens |

All adapters are implemented without a parameter-efficient fine-tuning library. Before every adapter experiment the following checks were run: LoRA and LoRMA+ leave the model output unchanged at initialisation (maximum difference below 1e-4), merged weights reproduce the adapter output, and prefix tuning matches a reference run in which the prefix tokens are actually prepended.

**Distillation recipe.** The student is trained in two stages from random weights: (1) language-model distillation on the Phase 1 packed blocks with the pretrained teacher evaluated live on each batch (4,000 steps of 32,768 tokens, about 131M tokens, learning rate 5e-4), and (2) NER distillation against the logits of the NER-fine-tuned teacher (learning rate 3e-4, up to 20 epochs). A student trained with the same settings but without a teacher serves as the baseline.

## 4. Submissions

| Submission | Phase 1 model | Technique |
| --- | --- | --- |
| 1 | GQA decoder | Full fine-tuning |
| 2 | Half-size GQA student | Distillation from submission 1 |
| 3 | Strided GQA decoder | LoRMA+ |
| 4 | Encoder-decoder (encoder only) | LoRA |

Further runs (full fine-tuning, LoRA, LoRMA+ and prefix tuning on the GQA, strided and encoder-decoder models, and the same set for the MLA decoder) were executed to compare techniques on the same backbone. Runs that differ only in the learning rate are tuning runs and not separate submissions.

## 5. Results

Test strict macro-F1 from the leaderboard (leaderboard best: 0.6801).

| Model | Technique | Test F1 | Prediction file(s) |
| --- | --- | --- | --- |
| GQA | Distilled student (best) | **0.4192** | `prediction_sub2_student_kd.json` |
| GQA | Full fine-tuning | 0.4005 | `prediction_sub1_gqa_full_ft.json` |
| GQA | LoRA | 0.3879 | `prediction_abl_ner_best_lora_lr*.json` |
| GQA | LoRMA+ | 0.3691 | `prediction_abl_ner_best_lorma_lr*.json` |
| GQA | Prefix tuning | 0.2817 | `prediction_abl_ner_best_prefix_lr*.json` |
| Strided | Full fine-tuning | 0.4098 | `prediction_second_full_ft.json` |
| Strided | LoRA (learning rate 0.003) | 0.3772 | `prediction_second_lora_lr0.003.json` |
| Strided | LoRA (learning rate 0.001) | 0.3638 | `prediction_second_lora_lr0.001.json` |
| Strided | LoRMA+ | 0.3758 | `prediction_second_lorma_lr*.json` |
| Strided | Prefix tuning | 0.2192 | `prediction_second_prefix_lr*.json` |
| Encoder-decoder | LoRA | 0.2633 | `prediction_encdec_lora_lr*.json` |
| Encoder-decoder | Full fine-tuning | 0.1853 | `prediction_encdec_full_ft.json` |
| Encoder-decoder | Prefix tuning | 0.0266 | `prediction_encdec_prefix_lr*.json` |
| MLA | Full fine-tuning | 0.2570 | `prediction_mla_full_ft.json` |
| MLA | LoRA | 0.2061 | `prediction_mla_lora_lr*.json` |
| MLA | LoRMA+ | 0.1809 | `prediction_mla_lorma_lr*.json` |

Precision and recall on the test set (as reported by the leaderboard):

| Model | Technique | Precision / recall |
| --- | --- | --- |
| GQA | Distilled student (best) | 0.3906 / 0.4688 |
| GQA | Full fine-tuning | 0.3517 / 0.4735 |
| GQA | LoRA | 0.3403 / 0.4594 |
| GQA | LoRMA+ | 0.3154 / 0.4538 |
| Strided | Full fine-tuning | 0.3777 / 0.4569 |
| Strided | LoRA (learning rate 0.003) | 0.3337 / 0.4421 |
| Strided | LoRMA+ | 0.3204 / 0.4877 |
| Strided | Prefix tuning | 0.1570 / 0.3840 |
| Encoder-decoder | LoRA | 0.2212 / 0.3370 |
| Encoder-decoder | Full fine-tuning | 0.1470 / 0.2622 |
| MLA | Full fine-tuning | 0.2453 / 0.2786 |

Files containing `lr*` exist for two learning rates; the reported score belongs to the variant that was submitted, except for the strided LoRA runs, where both learning rates were scored.

**Validation scores (GQA).** Full fine-tuning 0.3063 (87.5M trainable parameters, 1.3 min, 17.8 GB peak), distilled student 0.3219 (43.2M, 2.7 min, 13.9 GB), LoRMA+ 0.2923 (0.43M trainable, 0.49%) and LoRA 0.2629.

## 6. Ablations and observations

- **Lookahead.** Because the decoders are causal, letting each token's label be read from a later position (a shift of one or two tokens) was tested. No gain was observed, and the final setup uses no shift.
- **Class weights and regularisation.** Runs without class weights, with a lower learning rate (5e-5) and with stronger dropout (0.3) were compared; none improved on the baseline by more than the measurement noise (about 0.02).
- **Adapter learning rate.** LoRA and LoRMA+ at a learning rate of 3e-4 obtained 0.2349 and 0.2358, below the 1e-3 setting.
- **Strict versus lenient scoring.** Lenient F1 (about 0.52 to 0.56) is far above strict F1 (about 0.28 to 0.31), which indicates that entities are often located but their boundaries are inexact.
- **Overfitting.** The training loss falls to about 0.002 while the validation F1 stays flat, so the models overfit the 60 training documents quickly.
- **Distillation.** The student, with 49% of the parameters, matched the teacher on validation (0.322 against 0.306) and on test (0.419 against 0.401), although it is a weaker language model (validation perplexity 12.6 against 9.1).
- **Parameter-efficient methods.** LoRMA+ reached about 95% of the full fine-tuning validation score with 0.49% of the trainable parameters; prefix tuning was clearly the weakest technique.

## Code Organisation

The code of this phase is the section of each full notebook that starts at the heading of the phase. The extracted notebooks in `notebooks/` contain only that section and are intended for reading; the runnable versions are the full notebooks in `../notebooks/` (plain) and `../notebooks/annotated/` (with an explanation after every code cell).

| Model | Extract of this phase | Full notebook |
| --- | --- | --- |
| GQA decoder | `notebooks/gqa_phase2.ipynb` | `../notebooks/CS781_gqa_all_phases.ipynb` |
| Strided GQA decoder | `notebooks/strided_phase2.ipynb` | `../notebooks/CS781_strided_all_phases.ipynb` |
| MLA decoder | `notebooks/mla_phase2.ipynb` | `../notebooks/CS781_mla_all_phases.ipynb` |
| Encoder-decoder (6x6) | `notebooks/encdec_phase2.ipynb` | `../notebooks/CS781_encdec_all_phases.ipynb` |

The distilled student is trained inside this section (checkpoint `student_kd_stage1/best.pt`), and later phases reuse it.

## Execution Instructions

1. Open the relevant notebook in **Google Colab** with a GPU runtime (an A100 was used for all reported runs; an L4 with 24 GB is sufficient for the smaller runs but considerably slower).
2. Mount Google Drive when prompted. Outputs and checkpoints are written to `MyDrive/CS781_Capstone_Project_Phase_N`, where N is the phase number.
3. Add a Colab secret named `HF` that holds the access token of the course data repository. The configuration cell reads it as `HF_TOKEN`, and the repository identifier as `HF_DATASET_REPO`. The first cells of every notebook download the data.
4. Download the shared checkpoints (see the top-level README) into `MyDrive`, keeping the sub-folder names, or run the Phase 1 training cells to regenerate them.
5. Run the notebook **from top to bottom**. Later sections reuse the tokenizer, model classes and helper functions defined earlier, and stop with an assertion if one of them is missing.
6. Each phase section ends by writing `prediction*.json` files in the format expected by the leaderboard.
