# Phase 3: Rhetorical Role Prediction (RR)

The task is to label every sentence of a court judgment with one of 13 rhetorical roles. The principal finding of this phase is that reading a long document window, instead of one sentence at a time, is the largest single improvement; the best submission is an ensemble of GQA-family models.

## 1. Data and metric

- **Labels (13).** Fact, Issue, ArgumentPetitioner, ArgumentRespondent, Statute, Dissent, PrecedentReliedUpon, PrecedentNotReliedUpon, PrecedentOverruled, RulingByLowerCourt, RatioOfTheDecision, RulingByPresentCourt and None. Every sentence receives exactly one label.
- **Domains.** Corporate Law (CL) and Information Technology (IT).
- **Splits.** 40 training documents per domain (10,593 CL and 6,252 IT sentences), 5 development documents per domain (1,396 and 746 sentences) and 5 hidden test documents per domain (1,339 and 858 sentences).
- **Document length.** The median document has about 5,000 tokens and the longest CL training document about 71,000 tokens, so no document fits in one 1,024-token window.
- **Imbalance.** Fact and RatioOfTheDecision dominate (about 2,900 CL training sentences each), whereas Issue has 17 CL training sentences and PrecedentOverruled occurs three times in total. Some labels are absent from the development sets.
- **Metric.** Macro-F1 (secondary: macro-precision and macro-recall) computed by a scorer written for this project. Model selection used the official development files (10 documents), so development scores are noisy.

## 2. Methodology

1. **Sentences to tokens.** Each sentence is encoded with the Phase 1 tokeniser (capped at 128 tokens) and followed by a newline token that marks the end of the sentence.
2. **Windows of whole sentences.** Consecutive sentences are packed into windows of up to 1,024 tokens (1,536 and 2,048 in the longer-window runs). Windows overlap, each starting about 75% of the way through the previous one, and every sentence appears in at least one window.
3. **Sentence representation.** The backbone reads the window. For every sentence, the hidden state at the newline token (which has seen the whole sentence and, for causal decoders, everything before it) is concatenated with the mean of the token states of the sentence. Dropout (0.1) and a linear layer produce 13 scores.
4. **Context modes.** The effect of context is measured with three window types: `sentence` (one sentence alone), `local8` (up to eight consecutive sentences) and `document` (as many sentences as fit in the window).
5. **Training.** Class-weighted cross-entropy (square root of the inverse frequency, clipped at 10), AdamW (backbone 1e-4, head 1e-3 for full fine-tuning), 6% warm-up and linear decay, batches of 8 windows, bf16 and early stopping on development macro-F1. Both domains are trained together in one model and reported separately.
6. **Prediction.** Probabilities from overlapping windows are averaged and the arg-max is taken for each sentence.
7. **Encoder-decoder.** The encoder takes no attention mask, so each window is passed through it alone to prevent padding from leaking into other tokens.
8. **Additions.** Relative-position features (the position of a sentence in the document and the document length) and Viterbi decoding over the label sequence with transition probabilities estimated from the training labels were also evaluated.

## 3. Experiments

**Required experiments (four different Phase 1 models, techniques including LoRMA+ and knowledge distillation).**

| # | Phase 1 model | Technique | Configuration |
| --- | --- | --- | --- |
| 1 | GQA decoder | Full fine-tuning | All 87.5M parameters, 1,024-token document windows |
| 2 | Half-size GQA student (random initialisation) | Distillation from experiment 1 | 43.2M parameters (7 layers, d_model 576); language-model distillation on Phase 1 data, then task training with loss 0.5 CE + 0.5 T^2 KL (T = 2) |
| 3 | Strided GQA decoder | LoRMA+ | Rank 10, alpha 20 on the q, k, v and output projections of every layer, plus the head |
| 4 | Encoder-decoder (encoder only) | LoRA | Rank 8, alpha 16 on the attention projections of the encoder, plus the head (231,693 trainable parameters) |

**Additional runs.**

| Run | Model | Purpose |
| --- | --- | --- |
| local8, sentence | GQA, full fine-tuning | Effect of context |
| doc1536 | GQA, full fine-tuning | Effect of a longer window |
| student without teacher | Half-size student | Contribution of the teacher signal |
| position features | GQA | Value of document position |
| Viterbi decoding | GQA | Value of label-sequence smoothing |
| ensembles | GQA family | Value of combining runs |
| full, LoRA, LoRMA+ (r = 6) | MLA decoder | The same techniques on a fourth backbone |

**Ensembles.** The best submission, `ens_best_l0`, averages the class probabilities of three GQA-family runs: GQA with a 1,024-token window, GQA with a 2,048-token window and the half-size student trained without a teacher. This subset was the best of all subsets of size two or more on the 10 development documents, so its development score is optimistic. The suffix `l0` denotes plain arg-max decoding.

## 4. Results

Test macro-F1 from the leaderboard (leaderboard best: 0.6778).

| Model | Run | Test F1 | Prediction file |
| --- | --- | --- | --- |
| GQA family | Ensemble `ens_best_l0` (best) | **0.5018** | `prediction_ens_best_l0_all.json` |
| GQA | Full fine-tuning with Viterbi decoding | 0.4830 | `prediction_gqa_full_document_viterbi_all.json` |
| GQA | Position features with Viterbi decoding | 0.4796 | produced by the Viterbi section (see notebook) |
| GQA | Full fine-tuning, 1,024-token window | 0.4718 | `prediction_gqa_full_document_all.json` |
| GQA | Window of 1,536 tokens | 0.4715 | `prediction_gqa_full_doc1536_all.json` |
| GQA | Position features | 0.4652 | `prediction_gqa_full_document_pos_all.json` |
| GQA family | Student without teacher | 0.4605 | `prediction_student_nokd_document_all.json` |
| Strided | LoRMA+ | 0.4441 | `prediction_second_lorma_lr*_all.json` |
| GQA family | Student with distillation | 0.4271 | `prediction_student_kd_document_all.json` |
| GQA | Local context of eight sentences | 0.3732 | `prediction_gqa_full_local8_all.json` |
| GQA | Sentence only | 0.3659 | `prediction_gqa_full_sentence_all.json` |
| MLA | Full fine-tuning | 0.3589 | `prediction_mla_full_document_all.json` |
| Encoder-decoder | LoRA | 0.3508 | `prediction_encdec_lora_lr0.002_all.json` |
| MLA | LoRA | 0.2624 | `prediction_mla_lora_document_all.json` |
| MLA | LoRMA+ | 0.2283 | `prediction_mla_lorma_document_all.json` |

Precision and recall on the test set (as reported by the leaderboard):

| Model | Run | Precision / recall |
| --- | --- | --- |
| GQA family | Ensemble `ens_best_l0` (best) | 0.5039 / 0.5067 |
| GQA | Full fine-tuning with Viterbi decoding | 0.5011 / 0.4755 |
| GQA | Position features with Viterbi decoding | 0.5145 / 0.4667 |
| GQA | Full fine-tuning, 1,024-token window | 0.4696 / 0.4817 |
| Strided | LoRMA+ | 0.4959 / 0.4523 |
| GQA family | Student with distillation | 0.4767 / 0.4245 |
| GQA | Sentence only | 0.3789 / 0.3795 |
| MLA | Full fine-tuning | 0.3810 / 0.3577 |
| Encoder-decoder | LoRA | 0.3360 / 0.3903 |

Every run has `_all`, `_CL` and `_IT` versions in the notebook output; only the `_all` version is kept in this repository.

## 5. Ablations and observations

- **Context (development macro-F1, GQA with full fine-tuning).** Sentence only 0.350, eight sentences 0.440, 1,024-token window 0.537, 1,536-token window 0.538 and 2,048-token window 0.561. The gain held in both domains (CL 0.264, 0.323, 0.465; IT 0.379, 0.497, 0.570), and the shorter modes also scored lower on the hidden test set.
- **Distillation.** Development F1 of the teacher 0.5373, of the student with distillation 0.4810 and of the same student without a teacher 0.4773; the teacher signal therefore contributes little beyond the smaller model on this small dataset.
- **Position features and Viterbi decoding.** Position features alone slightly lowered the development score (0.529) but, combined with Viterbi decoding, reached 0.543.
- **Ensemble.** The selected three-run ensemble reached 0.572 on development.
- **Backbone.** The MLA decoder and the encoder-decoder trailed the GQA family; with only five development documents per domain, differences of about 0.02 or less are within noise.

## Code Organisation

The code of this phase is the section of each full notebook that starts at the heading of the phase. The extracted notebooks in `notebooks/` contain only that section and are intended for reading; the runnable versions are the full notebooks in `../notebooks/` (plain) and `../notebooks/annotated/` (with an explanation after every code cell).

| Model | Extract of this phase | Full notebook |
| --- | --- | --- |
| GQA decoder | `notebooks/gqa_phase3.ipynb` | `../notebooks/CS781_gqa_all_phases.ipynb` |
| Strided GQA decoder | `notebooks/strided_phase3.ipynb` | `../notebooks/CS781_strided_all_phases.ipynb` |
| MLA decoder | `notebooks/mla_phase3.ipynb` | `../notebooks/CS781_mla_all_phases.ipynb` |
| Encoder-decoder (6x6) | `notebooks/encdec_phase3.ipynb` | `../notebooks/CS781_encdec_all_phases.ipynb` |

The ensemble code is located in the Phase 3 section of the GQA notebook (the cell that builds the candidate list, searches all subsets and calls `write_ens`). It requires the saved probability files `probs_test_*.pkl` that are written when each run is trained; these files are not included in this repository.

## Execution Instructions

1. Open the relevant notebook in **Google Colab** with a GPU runtime (an A100 was used for all reported runs; an L4 with 24 GB is sufficient for the smaller runs but considerably slower).
2. Mount Google Drive when prompted. Outputs and checkpoints are written to `MyDrive/CS781_Capstone_Project_Phase_N`, where N is the phase number.
3. Add a Colab secret named `HF` that holds the access token of the course data repository. The configuration cell reads it as `HF_TOKEN`, and the repository identifier as `HF_DATASET_REPO`. The first cells of every notebook download the data.
4. Download the shared checkpoints (see the top-level README) into `MyDrive`, keeping the sub-folder names, or run the Phase 1 training cells to regenerate them.
5. Run the notebook **from top to bottom**. Later sections reuse the tokenizer, model classes and helper functions defined earlier, and stop with an assertion if one of them is missing.
6. Each phase section ends by writing `prediction*.json` files in the format expected by the leaderboard.
