# Phase 7: Prior Case Retrieval (PCR)

Given a query case and a pool of candidate cases, the task is to return the seven candidates that the query most likely cites. A bi-encoder with a learnable projection was trained with a contrastive objective on top of each Phase 1 backbone.

## 1. Data and metric

- **Sizes.** Training 827 queries (4,320 candidate documents), development 118 queries (1,023 candidates) and test 237 queries (1,727 candidates).
- **Gold citations.** On the training set a query has a mean of 6.42 gold candidates (median 4, maximum 52); on the development set the mean is 8.25 (median 5, maximum 92). Every gold identifier lies inside the pool of its query. The document of the query itself also appears in its pool and must be excluded from the ranking.
- **Overlap.** Candidate pools overlap across splits (436 documents shared by training and development, 740 by training and test, 220 by development and test), but no query is shared.
- **Length.** The median query has 5,863 tokens (90th percentile 17,328; maximum 61,879) and the median candidate 4,100 tokens (maximum 87,822).
- **Chance level.** A random ranking yields a Micro-F1@7 of only about 0.0016 (training) and 0.0074 (development).
- **Metric.** Micro-F1@7: the top seven candidates are returned for every query, all (query, candidate) decisions are pooled across queries, and F1 is computed from the total true positives, false positives and false negatives. Micro-precision and micro-recall at seven are reported alongside.

## 2. Methodology

A bi-encoder is used: queries and candidates are encoded independently into vectors and ranked by cosine similarity.

1. **Chunking.** A document is cut into chunks of 1,024 tokens, which handles documents far longer than the model context.
2. **Chunk feature.** Each chunk passes through the backbone and the mean over its token states is used. The last-token state is not used for retrieval, because it was much worse in the zero-shot test.
3. **Document vector.** The mean of all chunk vectors of a document is passed through a learnable linear projection whose weight is initialised to the identity, so that training starts from the zero-shot ranking of the frozen model.
4. **Scoring.** Cosine similarity between the query vector and every candidate vector in its pool; the query is excluded and the top seven candidates are returned.
5. **Objective.** InfoNCE with temperature 0.05. The positives of a query are its gold candidates; the negatives are the positives of the other queries in the same batch (in-batch negatives) plus one hard negative per query, taken from the 30 documents closest to the query under the frozen zero-shot model that are not gold.
6. **Training views.** To limit the cost, every document is represented by one random 512-token window per step. At inference, all chunks are used.
7. **Optimiser.** AdamW, weight decay 0.01, 6% warm-up followed by linear decay, learning rate 1e-3 for the projection and head and a smaller learning rate for the backbone or adapters (see below). Development Micro-F1@7 is computed periodically and the best checkpoint is retained.

**Zero-shot baseline.** For the frozen GQA model, mean-token pooling gave a development Micro-F1@7 of about 0.296 to 0.307, whereas last-token pooling reached at most 0.087.

**Distillation.** The teacher is the fine-tuned GQA retrieval model and the student is the 43.2M-parameter model. The student is trained to reproduce the softmax distribution of the teacher over the candidate scores of the batch, with loss 0.5 InfoNCE + 0.5 KL divergence. The teacher is evaluated online.

**Adapters and partial fine-tuning.** LoRMA+ is inserted into the attention projections of the strided model. For MLA and the encoder-decoder only the top layers are unfrozen (four for MLA, two for the encoder-decoder), because full fine-tuning at these lengths exceeds the available time and memory.

## 3. Experiments

| Experiment | Model | Technique | Learning rate | Steps | Batch |
| --- | --- | --- | --- | --- | --- |
| 0 (baseline) | Each of the four | Frozen, zero-shot pooling | none | none | none |
| 1 | GQA | Full fine-tuning, including the longer run `gqa_full_long` | see the GQA notebook | see the GQA notebook | see the GQA notebook |
| 2 | Strided | LoRMA+ (rank 64, alpha 128) | 5e-4 | 1,500 | 16 |
| 3 | MLA | Partial fine-tuning, top four layers | 2e-5 | 1,500 | 16 (8 if out of memory) |
| 4 | Encoder-decoder | Partial fine-tuning, top two layers | 5e-5 | 900 | 8 |
| 5 | Half-size student (43.2M) | Distillation from experiment 1 | 5e-5 | 1,500 | 16 |

The strided LoRMA+ run trains 3.21M parameters and the student 43.2M. The strided LoRMA+ run required 13.5 minutes and a peak of 18.2 GB. The MLA run used an L4 GPU with 24 GB and was much slower than on an A100.

## 4. Results

Test Micro-F1@7 from the leaderboard.

| Model | Technique | Micro-F1@7 | P@7 / R@7 | Prediction file |
| --- | --- | --- | --- | --- |
| Strided | LoRMA+ (best test score) | **0.3154** | 0.3219 / 0.3092 | `prediction_pcr_strided_lorma.json` |
| GQA | Full fine-tuning, longer run (selected on the leaderboard) | 0.3148 | 0.3213 / 0.3086 | `prediction_pcr_gqa_full_long.json` |
| GQA family | Distilled student | 0.3077 | 0.3140 / 0.3017 | `prediction_pcr_student_kd.json` |
| GQA | Full fine-tuning | 0.3006 | 0.3068 / 0.2947 | `prediction_pcr_gqa_full.json` |
| Encoder-decoder | Partial (top two layers) | 0.2233 | 0.2278 / 0.2189 | `prediction_pcr_encdec_partial.json` |
| MLA | Partial (top four layers) | 0.0685 | 0.0699 / 0.0672 | `prediction_pcr_mla_partial.json` |

The development Micro-F1@7 of the strided LoRMA+ run was 0.3446.

## 5. Observations

- The two best test scores differ by 0.0006, which is within the noise level on 237 queries.
- The MLA score is far below the other runs and is probably caused by a failed run rather than by a property of the architecture.
- LoRMA+ on the strided model (3.21M trainable parameters) matched full fine-tuning of the GQA model, which makes it the most parameter-efficient result of this phase.

## Code Organisation

The code of this phase is the section of each full notebook that starts at the heading of the phase. The extracted notebooks in `notebooks/` contain only that section and are intended for reading; the runnable versions are the full notebooks in `../notebooks/` (plain) and `../notebooks/annotated/` (with an explanation after every code cell).

| Model | Extract of this phase | Full notebook |
| --- | --- | --- |
| GQA decoder | `notebooks/gqa_phase7.ipynb` | `../notebooks/CS781_gqa_all_phases.ipynb` |
| Strided GQA decoder | `notebooks/strided_phase7.ipynb` | `../notebooks/CS781_strided_all_phases.ipynb` |
| MLA decoder | `notebooks/mla_phase7.ipynb` | `../notebooks/CS781_mla_all_phases.ipynb` |
| Encoder-decoder (6x6) | `notebooks/encdec_phase7.ipynb` | `../notebooks/CS781_encdec_all_phases.ipynb` |

## Execution Instructions

1. Open the relevant notebook in **Google Colab** with a GPU runtime (an A100 was used for all reported runs; an L4 with 24 GB is sufficient for the smaller runs but considerably slower).
2. Mount Google Drive when prompted. Outputs and checkpoints are written to `MyDrive/CS781_Capstone_Project_Phase_N`, where N is the phase number.
3. Add a Colab secret named `HF` that holds the access token of the course data repository. The configuration cell reads it as `HF_TOKEN`, and the repository identifier as `HF_DATASET_REPO`. The first cells of every notebook download the data.
4. Download the shared checkpoints (see the top-level README) into `MyDrive`, keeping the sub-folder names, or run the Phase 1 training cells to regenerate them.
5. Run the notebook **from top to bottom**. Later sections reuse the tokenizer, model classes and helper functions defined earlier, and stop with an assertion if one of them is missing.
6. Each phase section ends by writing `prediction*.json` files in the format expected by the leaderboard.
