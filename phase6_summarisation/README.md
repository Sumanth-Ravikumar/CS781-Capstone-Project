# Phase 6: Abstractive Legal Summarisation

The task is to generate a summary of an Indian court judgment. The highest BERTScore F1 was obtained by the GQA decoder with LoRMA+ adaptation, which matched the full fine-tuning result with only 2.6M trainable parameters.

## 1. Data and metric

- **Task.** Given the text of a judgment, generate one summary. The primary metric is BERTScore F1, and ROUGE-1 F1 and ROUGE-L F1 are secondary. The competition computes all three against private reference summaries. The submission is a JSON list of objects with `id` and `summary`, one per test document, uploaded as `prediction.json`.
- **Split.** The training file contains 7,030 examples and the test file 100 documents. No development split is provided; two training examples with empty summaries were dropped, 300 of the remaining 7,028 were held out at random (fixed seed) for validation, and the other 6,728 were used for training.
- **Length (tokens, Phase 1 tokeniser).** Training documents: median 3,699, 90th percentile 9,867, maximum 51,763. Test documents: median 3,754, maximum 45,404. Reference summaries: median 773, 90th percentile 1,833, maximum 9,803. Only 2.4% of training documents fit in 1,024 tokens and 55.8% in 4,096; only 26.0% of the summaries fit in 512 tokens.

## 2. Methodology

The decoders (GQA, strided, MLA) and the encoder-decoder are trained to write the summary from a shortened view of the judgment, because none of the models can read the whole document.

1. **Tokenisation.** Documents and summaries are tokenised once. For every document, the first 4,096 and the last 2,048 tokens and the full length are stored; for every training summary, its first 1,024 tokens.
2. **View.** The decoders read the first 1,024 and the last 480 tokens of a document (1,504 in total, or the whole document when shorter). The encoder-decoder reads the first 768 and the last 256 tokens, which fills its 1,024 source positions.
3. **Training sequence.** A decoder reads the view, then the text `SUMMARY:` on a new line, then the summary and the end-of-text token. The loss is computed on the summary tokens only. The summary is cut to its first 512 tokens (320 for the encoder-decoder, which is the decoder limit), and the end token is added only when the whole summary fits.
4. **Optimisation.** Cross-entropy on summary tokens, AdamW with weight decay 0.01, 6% warm-up followed by linear decay, bf16, gradient clipping at 1.0 and a batch of 8 sequences (MLA: 2 sequences with 4 accumulation steps). Validation loss on the 300 held-out documents is evaluated twice per epoch, the best checkpoint is retained and training stops after three evaluations without improvement.
5. **Generation.** Greedy decoding with a key-value cache and a block on the repetition of any 4-gram. Generation stops at the end-of-text token or at the length cap (512 new tokens for the decoders and 320 for the encoder-decoder).
6. **Local scoring.** ROUGE-1 and ROUGE-L F1 (the `rouge_score` package with stemming) on the first 40 held-out documents, against references cut to their first 1,024 tokens. This is a proxy and not the competition metric. BERTScore is not computed locally because it requires a pretrained encoder, whereas the project rules allow no pretrained weights. Checkpoints are selected by validation loss, not by ROUGE.

**MLA context.** The MLA model supports 4,096 positions, but a longer view (2,048 and 1,024 document tokens, 960 summary tokens) exhausted the memory of a 40 GB GPU. The MLA runs therefore use the same view as the other decoders with a smaller batch.

**Adapters.** LoRA and LoRMA+ use the implementations described in Phase 4, with rank 64 and alpha 128 on the q, k, v and output projections of every layer (for MLA, on its own projections). The Houlsby-style adapter is a bottleneck of width 128 (down-projection, GELU, up-projection, added back to the input) placed after the attention output projection and after the feed-forward output projection of every layer; its up-projection starts at zero, so that the model begins unchanged.

**Distillation.** The student is the half-size GQA-family model (43.2M parameters, 7 layers, d_model 576), initialised from the Phase 2 stage-1 checkpoint. Teacher and student read the same sequence, and the student is trained on the summary positions with loss 0.5 CE + 0.5 T^2 KL (T = 2) between the next-token distributions. The teacher runs alongside the student during training, so no teacher outputs are stored.

## 3. Experiments

| # | Phase 1 model | Technique | Configuration |
| --- | --- | --- | --- |
| 1 | GQA decoder | LoRMA+ | Rank 64, alpha 128 on q, k, v and output projections of all 10 layers (2.62M trainable of 90.1M), learning rate 5e-4 |
| 2 | Strided GQA decoder | LoRA | Rank 64, alpha 128 on the same projections (3.28M trainable of 90.8M), learning rate 5e-4 |
| 3 | MLA decoder | Houlsby-style bottleneck adapter | Width 128 after the attention and feed-forward output projections of all layers (3.95M trainable of 91.0M), learning rate 5e-4 |
| 4 | Encoder-decoder | Full fine-tuning | All parameters, learning rate 5e-5 |
| 5 | Half-size GQA-family student | Knowledge distillation | 43.2M parameters trained in full, learning rate 1e-4 |

Each method was run with one setting that was chosen in advance and not searched (5e-4 for the adapters, 5e-5 for full fine-tuning, 1e-4 for the student). Additional runs: GQA full fine-tuning, GQA LoRA, GQA Houlsby-style adapter, strided LoRMA+ and MLA LoRMA+.

## 4. Results

Test scores from the competition leaderboard.

| Model | Technique | BERTScore F1 | ROUGE-1 | ROUGE-L | Prediction file |
| --- | --- | --- | --- | --- | --- |
| GQA | LoRMA+ (best, submitted) | **0.8423** | 0.4322 | 0.2367 | `prediction_gqa_lorma.json` |
| GQA | Full fine-tuning | 0.8412 | 0.4406 | 0.2380 | `prediction_gqa_full.json` |
| GQA | LoRA | 0.8402 | 0.4367 | 0.2327 | `prediction_gqa_lora.json` |
| GQA | Houlsby-style adapter | 0.8399 | 0.4388 | 0.2375 | `prediction_gqa_houlsby.json` |
| GQA family | Distilled student | 0.8377 | 0.4276 | 0.2279 | `prediction_gqa_student_kd.json` |
| Strided | LoRMA+ | 0.8255 | 0.3601 | 0.2004 | `prediction_strided_lorma.json` |
| Encoder-decoder | Full fine-tuning | 0.8255 | 0.3494 | 0.1975 | `prediction_encdec_full.json` |
| Strided | LoRA | 0.8254 | 0.3664 | 0.2023 | `prediction_strided_lora.json` |
| MLA | Houlsby-style adapter | 0.7914 | 0.2854 | 0.1833 | `prediction_mla_houlsby.json` |
| MLA | LoRMA+ | 0.7886 | 0.3399 | 0.2031 | `prediction_mla_lorma.json` |

The strided LoRMA+ and encoder-decoder full fine-tuning runs both round to 0.8255 BERTScore F1, so the pairing of these two rows follows the order in which the scores were recorded.

## 5. Observations

- On the GQA decoder, the four adaptation methods (LoRMA+, full fine-tuning, LoRA and Houlsby-style adapter) lie within 0.002 of one another in BERTScore F1; LoRMA+ matches full fine-tuning with 2.6M trainable parameters.
- The GQA family is clearly ahead of the strided, encoder-decoder and MLA models in BERTScore and ROUGE.
- The student retained most of the quality of the GQA models (0.8377) with about half the parameters.

## Code Organisation

The code of this phase is the section of each full notebook that starts at the heading of the phase. The extracted notebooks in `notebooks/` contain only that section and are intended for reading; the runnable versions are the full notebooks in `../notebooks/` (plain) and `../notebooks/annotated/` (with an explanation after every code cell).

| Model | Extract of this phase | Full notebook |
| --- | --- | --- |
| GQA decoder | `notebooks/gqa_phase6.ipynb` | `../notebooks/CS781_gqa_all_phases.ipynb` |
| Strided GQA decoder | `notebooks/strided_phase6.ipynb` | `../notebooks/CS781_strided_all_phases.ipynb` |
| MLA decoder | `notebooks/mla_phase6.ipynb` | `../notebooks/CS781_mla_all_phases.ipynb` |
| Encoder-decoder (6x6) | `notebooks/encdec_phase6.ipynb` | `../notebooks/CS781_encdec_all_phases.ipynb` |

## Execution Instructions

1. Open the relevant notebook in **Google Colab** with a GPU runtime (an A100 was used for all reported runs; an L4 with 24 GB is sufficient for the smaller runs but considerably slower).
2. Mount Google Drive when prompted. Outputs and checkpoints are written to `MyDrive/CS781_Capstone_Project_Phase_N`, where N is the phase number.
3. Add a Colab secret named `HF` that holds the access token of the course data repository. The configuration cell reads it as `HF_TOKEN`, and the repository identifier as `HF_DATASET_REPO`. The first cells of every notebook download the data.
4. Download the shared checkpoints (see the top-level README) into `MyDrive`, keeping the sub-folder names, or run the Phase 1 training cells to regenerate them.
5. Run the notebook **from top to bottom**. Later sections reuse the tokenizer, model classes and helper functions defined earlier, and stop with an assertion if one of them is missing.
6. Each phase section ends by writing `prediction*.json` files in the format expected by the leaderboard.
