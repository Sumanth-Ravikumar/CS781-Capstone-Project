# Phase 5: Legal Statute Identification (LSI)

The task is multi-label classification: given a judgment, predict which of 100 statutes (masked identifiers 0 to 99) are cited. Full fine-tuning of the GQA model with per-label decision thresholds obtained the best score.

## 1. Data and metric

- **Labels.** 100 statutes; every identifier occurs in both the training and the development sets. Statute names are masked, so the model sees only identifiers and must learn them from the text. Entities and sections in the text are masked with tags such as `<SECTION>`, `<ACT>` and `<ENTITY>`.
- **Sizes.** Training 42,750 documents, development 10,181 and test 13,019 (identifiers and text only).
- **Imbalance.** The rarest statute has 272 training positives, the median 679 and the most frequent (identifier 5) 11,731. The mean number of labels per document is 3.71 (median 3, maximum 31); 21.4% of documents have a single label.
- **Length.** Median 663 tokens; 89.2% of documents fit in 4,096 tokens and 97.3% in 8,192.
- **Metrics.** Macro-F1 (primary), micro-F1 and subset accuracy.

## 2. Methodology

1. **Tokenisation.** Documents are tokenised with the Phase 1 tokeniser and the token identifiers are cached on Drive for reuse by every run.
2. **Document view.** The default view `ht4` retains the head and the tail of the document up to 4,096 tokens (the middle is dropped).
3. **Backbone.** The Phase 1 model is loaded from its checkpoint without the language-model head. The last-token state and the mean over all tokens are concatenated (width 2d).
4. **Head.** `Linear(2d, 100)` with one logit per statute. The bias is initialised to the log prior odds of each statute, so that the model predicts the base rates at step 0.
5. **Loss.** Binary cross-entropy per label with `pos_weight` = clip(sqrt(neg/pos), 1, 10), which prevents the model from always rejecting rare statutes without letting them dominate.
6. **Decision thresholds.** Probabilities are converted to labels with either one **global** threshold (grid from 0.05 to 0.775 in steps of 0.025, selected on development) or a separate **per-label** threshold tuned on held-out halves of the development set. Every document is forced to receive at least one label (its most probable statute).
7. **Recipe.** AdamW, weight decay 0.01, head learning rate 1e-3, backbone learning rate 5e-5 for full fine-tuning, 6% warm-up followed by linear decay, bf16, gradient clipping at 1.0, two epochs, evaluation twice per epoch and early stopping with a patience of three evaluations. A five-epoch GQA run overfitted and scored lower, so two epochs were used.

**Distillation.** The teacher is the fully fine-tuned GQA model (87.7M). The student (43.2M, d_model 576, 7 layers, initialised from the Phase 2 stage-1 checkpoint) is trained on the labels together with the soft targets of the teacher.

**Adapters.** LoRA and LoRMA+ (see Phase 2 and Phase 4) are inserted into the attention projections (for MLA: `q_up_proj`, `kv_down_proj`, `kv_up_proj`, `out_proj`). Only the adapter parameters and the head are trained. LoRMA+ is the identity at initialisation (verified, difference 0.0).

## 3. Experiments

| Experiment | Model | Technique | Purpose |
| --- | --- | --- | --- |
| 1 | GQA decoder | Full fine-tuning (87.7M trainable) | Reference model and distillation teacher |
| 2 | Strided GQA decoder | LoRMA+ | Required technique on a sparse-attention model |
| 3 | MLA decoder | Full fine-tuning and partial fine-tuning (top four layers) | Cost and quality of a compressed key-value model |
| 4 | Encoder-decoder | Full fine-tuning | Usefulness of an encoder-decoder document representation |
| 5 | Half-size student (43.3M) | Distillation from experiment 1 | Distillation with a smaller student |

## 4. Results

Test scores are taken from the Codabench leaderboard.

| Model | Technique | Macro-F1 (per-label) | Macro-F1 (global) | Prediction file |
| --- | --- | --- | --- | --- |
| GQA | Full fine-tuning (best) | **0.3867** | 0.3838 | `prediction_gqa_full_ht4_perlabel.json` (`_global`) |
| GQA family | Distilled student | 0.3856 | 0.3809 | `prediction_student_kd_ht4_*.json` |
| Encoder-decoder | Full fine-tuning | 0.3363 | 0.3291 | `prediction_encdec_full_ht4_*.json` |
| MLA | Full fine-tuning | 0.3271 | 0.3193 | `prediction_mla_full_ht4_*.json` |
| MLA | Partial (top four layers) | 0.1925 | 0.1815 | `prediction_mla_partial4_ht4_*.json` |
| Strided | LoRMA+ | 0.1253 | 0.1043 | `prediction_second_lorma_ht4_*.json` |

Micro-F1 and subset accuracy on the test set (per-label thresholds):

| Model | Technique | Micro-F1 | Subset accuracy |
| --- | --- | --- | --- |
| GQA | Full fine-tuning (best) | 0.4833 | 0.0424 |
| GQA family | Distilled student | 0.4819 | 0.0432 |
| Encoder-decoder | Full fine-tuning | 0.4362 | 0.0226 |
| MLA | Full fine-tuning | 0.4238 | 0.0203 |
| Strided | LoRMA+ | 0.1815 | 0.0000 |

The submissions of this phase were made after the Phase 5 deadline.

**Development scores (global threshold).** GQA full 0.3886 and student 0.3835. The learning curve of the GQA model (development macro-F1 at the best global threshold) was 0.3003 after about one quarter of training, 0.3456 at one half, 0.3648 at three quarters and 0.3886 at the end; the student followed the same shape (0.3123, 0.3496, 0.3644, 0.3835).

## 5. Observations

- Per-label thresholds improved the test macro-F1 for all six models for which both variants were scored, at the cost of slightly lower subset accuracy. A global threshold is nearly as effective as the 0.5 default because the weighted loss already centres the probabilities.
- The 43.3M student retained about 98.7% of the development macro-F1 of its teacher (0.3835 against 0.3886) with roughly half of the parameters.
- Macro-F1 is dominated by the rare statutes; the prior-odds bias and the weighted loss were used to prevent collapse to the frequent statutes.
- Cost: GQA full fine-tuning required about 28 minutes of training and evaluation on a 40 GB A100, the student about 19 minutes.
- Limitations: masked statute names remove a useful cue, documents longer than 4,096 tokens lose their middle part, and the development set was used to select thresholds and checkpoints, so development scores are slightly optimistic.

## Code Organisation

The code of this phase is the section of each full notebook that starts at the heading of the phase. The extracted notebooks in `notebooks/` contain only that section and are intended for reading; the runnable versions are the full notebooks in `../notebooks/` (plain) and `../notebooks/annotated/` (with an explanation after every code cell).

| Model | Extract of this phase | Full notebook |
| --- | --- | --- |
| GQA decoder | `notebooks/gqa_phase5.ipynb` | `../notebooks/CS781_gqa_all_phases.ipynb` |
| Strided GQA decoder | `notebooks/strided_phase5.ipynb` | `../notebooks/CS781_strided_all_phases.ipynb` |
| MLA decoder | `notebooks/mla_phase5.ipynb` | `../notebooks/CS781_mla_all_phases.ipynb` |
| Encoder-decoder (6x6) | `notebooks/encdec_phase5.ipynb` | `../notebooks/CS781_encdec_all_phases.ipynb` |

## Execution Instructions

1. Open the relevant notebook in **Google Colab** with a GPU runtime (an A100 was used for all reported runs; an L4 with 24 GB is sufficient for the smaller runs but considerably slower).
2. Mount Google Drive when prompted. Outputs and checkpoints are written to `MyDrive/CS781_Capstone_Project_Phase_N`, where N is the phase number.
3. Add a Colab secret named `HF` that holds the access token of the course data repository. The configuration cell reads it as `HF_TOKEN`, and the repository identifier as `HF_DATASET_REPO`. The first cells of every notebook download the data.
4. Download the shared checkpoints (see the top-level README) into `MyDrive`, keeping the sub-folder names, or run the Phase 1 training cells to regenerate them.
5. Run the notebook **from top to bottom**. Later sections reuse the tokenizer, model classes and helper functions defined earlier, and stop with an assertion if one of them is missing.
6. Each phase section ends by writing `prediction*.json` files in the format expected by the leaderboard.
