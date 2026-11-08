# Phase 4: Court Judgment Prediction (CJPE)

The task is to predict the outcome of an Indian court case (binary label) from the text of the judgment. Full fine-tuning of the GQA model and an ensemble of the GQA model with the distilled student produced the strongest results.

## 1. Data and metric

- **Splits.** Training 37,387 cases (59% label 0), development 3,505 (almost balanced) and test 1,517 (labels private; only identifiers and text are read).
- **Length.** Judgments are long (median about 3,900 tokens, maximum above 120,000) and 30.6% of training cases exceed 4,096 tokens.
- **Metric.** Macro-F1 (secondary: precision and recall). The submission is a JSON list of `{"id", "pred"}` objects.

## 2. Methodology

A judgment is cut into chunks of 1,024 tokens. Every chunk is encoded by the Phase 1 model, the chunk vectors are averaged into one document vector, and a linear layer predicts the label. The pipeline is identical for the four Phase 1 architectures, so experiments differ only in the model and the adaptation method.

1. **Tokenisation.** Every case is tokenised once; the first 4,096 and the last 4,096 tokens are cached on Drive and shared by all runs.
2. **View selection.** A view defines which chunks are read (table below).
3. **Chunk encoding.** Each chunk (at most 1,024 tokens) is represented by a vector of size 2d: the hidden state of its final token concatenated with the mean of all its token states.
4. **Pooling.** The chunk vectors of a case are averaged, followed by dropout (0.1) and a newly initialised `Linear(2d, 2)` head.
5. **Training.** Class-weighted cross-entropy (weights n / (2 x class count), about 0.85 for label 0 and 1.22 for label 1), AdamW with weight decay 0.01, 6% warm-up followed by linear decay, bf16, gradient clipping at 1.0 and head learning rate 1e-3. A batch holds 16 chunks (four cases for the four-chunk views). Training runs for two epochs, development is evaluated every half epoch, and the checkpoint with the best development macro-F1 is retained (patience of three evaluations).
6. **Prediction.** The label is the arg-max of the two logits (threshold 0.5).

| View | Content |
| --- | --- |
| `first1` | First 1,024 tokens |
| `last1` | Last 1,024 tokens |
| `first4` | First 4,096 tokens as up to four chunks |
| `last4` | Last 4,096 tokens as up to four chunks aligned to the end of the judgment |
| `ht4` | First 2,048 and last 2,048 tokens as four chunks |

The main runs use `last4`, because the reasoning and the decision of a judgment are located near its end. For cases longer than 4,096 tokens, no view reads the middle of the judgment.

**Distillation.** The teacher is the fine-tuned GQA model. Its logits for all 37,387 training cases are computed once on the `last4` view and saved. The student then trains offline against the stored logits with loss 0.5 CE + 0.5 T^2 KL (T = 2), so no teacher is required in memory during student training.

**Adapters.** LoRA and LoRMA+ are implemented from scratch. LoRA computes W0 x + (alpha/r) B(Ax). LoRMA+ computes y + (alpha/r) B(Ay) with y = W0 x, which corresponds to ((alpha/r) B A + I) W0; it starts as the identity and can be merged into the base weight. Both are verified to leave the model output unchanged at initialisation. In every adapter run the Phase 1 weights are frozen, and only the adapters and the new head are trained.

**Encoder-decoder.** Only the encoder is used. Each chunk passes through the encoder on its own, because the encoder takes no attention mask; features and pooling match the decoders.

## 3. Experiments

| # | Phase 1 model | Technique | Configuration |
| --- | --- | --- | --- |
| 1 | GQA decoder | Full fine-tuning | All 87.5M parameters, backbone learning rate 5e-5 |
| 2 | Half-size GQA-family student | Distillation from experiment 1 | 43.2M backbone (7 layers, d_model 576), initialised from the Phase 2 stage-1 checkpoint, backbone learning rate 2e-4 |
| 3 | MLA decoder | LoRMA+ | Rank 6, alpha 12 on the q-up, kv-down, kv-up and output projections, plus the head (0.49M trainable of 87.5M), learning rate 3e-4 |
| 4 | Encoder-decoder (encoder only) | LoRA | Rank 8, alpha 16 on the q, k, v and output projections of the encoder, plus the head (0.22M trainable of 84.1M), learning rate 2e-3 |

Each method was run with one setting and was not tuned. One additional run, full fine-tuning of the strided model on the `last4` view, was submitted after the deadline.

**Ensemble.** `ens_best` averages the class probabilities of the GQA model with full fine-tuning and the distilled student (both on `last4`). It was the best subset by development macro-F1 (0.7252 against 0.7170 for the GQA model alone).

## 4. Results

Test macro-F1 from the leaderboard. The precision and recall columns are the values reported by the leaderboard.

| Model | Technique | Test F1 | P / R | Prediction file |
| --- | --- | --- | --- | --- |
| GQA family | Ensemble of GQA full and student (best) | **0.7229** | 0.7395 / 0.6929 | `prediction_ens_best.json` |
| GQA | Full fine-tuning | 0.7157 | 0.7478 / 0.6575 | `prediction_gqa_full_last4.json` |
| Strided | Full fine-tuning (submitted after the deadline) | 0.6766 | 0.7460 / 0.5551 | `prediction_second_full_last4.json` |
| GQA family | Distilled student | 0.5205 | 0.5312 / 0.7493 | `prediction_student_kd_last4.json` |
| MLA | LoRMA+ | 0.5062 | 0.5102 / 0.5892 | `prediction_mla_lorma_last4.json` |
| Encoder-decoder | LoRA | 0.3676 | 0.4730 / 0.0459 | `prediction_encdec_lora_last4.json` |

**Compute (single A100, 80 GB).** GQA full fine-tuning: 56.9 minutes, 37.0 GB peak. Student: 37.6 minutes, 30.3 GB. MLA LoRMA+: 57.5 minutes, 21.0 GB.

## 5. Observations

- Full fine-tuning of the GQA model was the only single run that trained to a high score. The distilled student trained but remained well below its teacher, and the three adapter runs on other backbones finished close to or below chance level on development. The comparison is not controlled, because full fine-tuning was run on the GQA model only and the adapters on other models.
- The encoder-decoder LoRA run predicts almost exclusively one class (recall of 0.0459). 
- Class imbalance was handled with class weights and was not further tuned.
- The adapter layers themselves behave correctly (identity at initialisation, mergeable into the weights).

## Code Organisation

The code of this phase is the section of each full notebook that starts at the heading of the phase. The extracted notebooks in `notebooks/` contain only that section and are intended for reading; the runnable versions are the full notebooks in `../notebooks/` (plain) and `../notebooks/annotated/` (with an explanation after every code cell).

| Model | Extract of this phase | Full notebook |
| --- | --- | --- |
| GQA decoder | `notebooks/gqa_phase4.ipynb` | `../notebooks/CS781_gqa_all_phases.ipynb` |
| Strided GQA decoder | `notebooks/strided_phase4.ipynb` | `../notebooks/CS781_strided_all_phases.ipynb` |
| MLA decoder | `notebooks/mla_phase4.ipynb` | `../notebooks/CS781_mla_all_phases.ipynb` |
| Encoder-decoder (6x6) | `notebooks/encdec_phase4.ipynb` | `../notebooks/CS781_encdec_all_phases.ipynb` |

The ensemble cell (Phase 4 section of the GQA notebook) loads `dev_logits_*.npy` and `test_logits_*.npy` from `MyDrive/CS781_Capstone_Project_Phase_4/`; these files are written by the training runs and are not included in this repository.

## Execution Instructions

1. Open the relevant notebook in **Google Colab** with a GPU runtime (an A100 was used for all reported runs; an L4 with 24 GB is sufficient for the smaller runs but considerably slower).
2. Mount Google Drive when prompted. Outputs and checkpoints are written to `MyDrive/CS781_Capstone_Project_Phase_N`, where N is the phase number.
3. Add a Colab secret named `HF` that holds the access token of the course data repository. The configuration cell reads it as `HF_TOKEN`, and the repository identifier as `HF_DATASET_REPO`. The first cells of every notebook download the data.
4. Download the shared checkpoints (see the top-level README) into `MyDrive`, keeping the sub-folder names, or run the Phase 1 training cells to regenerate them.
5. Run the notebook **from top to bottom**. Later sections reuse the tokenizer, model classes and helper functions defined earlier, and stop with an assertion if one of them is missing.
6. Each phase section ends by writing `prediction*.json` files in the format expected by the leaderboard.
