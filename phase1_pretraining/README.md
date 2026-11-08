# Phase 1: Pretraining of Legal Language Models

Four language models of approximately 85 million parameters were pretrained from scratch on Indian legal text: a dense grouped-query-attention (GQA) decoder, a strided-attention GQA decoder, a DeepSeek-style multi-head latent attention (MLA) decoder and an encoder-decoder. The dense GQA decoder obtained the best leaderboard score and was used as the primary backbone in the subsequent phases.

## 1. Data

- **Corpus.** Indian legal text (acts and Supreme Court judgments). Tribunal documents were dropped to reduce preprocessing time.
- **Packing.** The corpus is tokenised and packed into fixed-length blocks of 2,048 tokens (about 390 million training tokens), stored as training and validation splits that are reused by every experiment.
- **Test.** The leaderboard evaluates 1,000 generated continuations with BERTScore F1 (primary), ROUGE-L and BLEU-4.

## 2. Tokeniser

A byte-level byte-pair-encoding (BPE) tokeniser with a vocabulary of 32,000 tokens was implemented from scratch (class `ByteBPE` and function `train_bpe` in `tokenizer/manual_bpe.py`) and trained on the Phase 1 corpus. The trained vocabulary is stored in `tokenizer/legal_bpe_32k_FINAL.json`. The special tokens are `<|endoftext|>` (identifier 0) and `[PAD]` (identifier 1). The same file is used in every subsequent phase.

## 3. Architectures

All decoders share the same building blocks, implemented in PyTorch: RMSNorm, rotary position embeddings (RoPE), SwiGLU feed-forward layers and tied input and output embeddings. Attention is computed with explicit matrix products, masks and softmax together with a key-value cache. Neither fused attention kernels nor pretrained library models are used.

| Model | Attention | Configuration | Parameters |
| --- | --- | --- | --- |
| GQA decoder | Grouped-query attention, 12 query heads sharing 4 key-value heads | 10 layers, d_model 768, context 2,048 | about 87.5M |
| Strided GQA decoder | GQA restricted to a local window of 256 positions plus a strided set of positions (stride 32 in even layers, 128 in odd layers) | 10 layers, d_model 768, context 2,048 | about 87.5M |
| MLA decoder | DeepSeek-style multi-head latent attention: compressed key-value latent of rank 128, query latent of rank 256, RoPE applied to a 64-dimensional part | 10 layers, d_model 768, 12 heads, context 4,096 | about 87M |
| Encoder-decoder (6x6) | Bidirectional encoder, causal decoder with cross-attention, 10 heads with 5 key-value heads | 6 encoder and 6 decoder layers, d_model 640 | about 83.9M |

The sparse attention pattern of the strided model introduces no additional parameters. The MLA configuration was reduced from an initial design of 154.7M parameters to fit the parameter budget.

## 4. Training

- **Objective.** Next-token prediction with cross-entropy for the decoders. The encoder-decoder uses a packed source-to-target objective (source of about 1,024 tokens, target of about 256 tokens, loss on the target only).
- **Optimisation.** AdamW with warm-up followed by linear decay, bf16 precision and gradient accumulation to fit a single A100 GPU.
- **GQA decoder.** 20,000 optimiser steps; the validation loss decreased from 4.37 to 2.19 (perplexity of about 9) and was still decreasing at the end, with no sign of overfitting. A later re-evaluation on held-out blocks gave 2.19 to 2.21.
- **Strided GQA decoder.** Validation loss of about 2.35 on 32 held-out blocks, which is higher than the dense model, as expected for restricted attention.
- **Verification.** Decoding with and without the key-value cache agrees to numerical precision (verified by a built-in check). Simulated INT8 weight quantisation left the validation loss essentially unchanged (2.1879 to 2.1883).

## 5. Results

| Model | Test BERTScore F1 | ROUGE-L | BLEU-4 | Prediction file |
| --- | --- | --- | --- | --- |
| GQA decoder (best) | **0.8342** | 0.2010 | 0.0581 | `prediction.json` |
| Strided GQA decoder | 0.8312 | 0.1967 | 0.0522 | `prediction_alt_stride_32_128.json` |
| Encoder-decoder (6x6) | 0.8215 | 0.1816 | 0.0292 | `prediction_encdec_6x6.json` |
| MLA decoder | 0.8064 | 0.1775 | 0.0287 | not included (see notebook output) |

## Code Organisation

The code of this phase is the section of each full notebook that starts at the heading of the phase. The extracted notebooks in `notebooks/` contain only that section and are intended for reading; the runnable versions are the full notebooks in `../notebooks/` (plain) and `../notebooks/annotated/` (with an explanation after every code cell).

| Model | Extract of this phase | Full notebook |
| --- | --- | --- |
| GQA decoder | `notebooks/gqa_phase1.ipynb` | `../notebooks/CS781_gqa_all_phases.ipynb` |
| Strided GQA decoder | `notebooks/strided_phase1.ipynb` | `../notebooks/CS781_strided_all_phases.ipynb` |
| MLA decoder | `notebooks/mla_phase1.ipynb` | `../notebooks/CS781_mla_all_phases.ipynb` |
| Encoder-decoder (6x6) | `notebooks/encdec_phase1.ipynb` | `../notebooks/CS781_encdec_all_phases.ipynb` |

## Artefacts

The tokenised dataset is too large for this repository and is regenerated by the notebook (sections on tokenisation and packing). The checkpoints are shared through the link in the top-level README and are written to `MyDrive/CS781_Capstone_Project_Phase_1/` (`gqa_checkpoints/`, `alt_stride_32_128_checkpoints/`, `encdec_6x6_checkpoints/`, `DeepSeek/`).

## Execution Instructions

1. Open the relevant notebook in **Google Colab** with a GPU runtime (an A100 was used for all reported runs; an L4 with 24 GB is sufficient for the smaller runs but considerably slower).
2. Mount Google Drive when prompted. Outputs and checkpoints are written to `MyDrive/CS781_Capstone_Project_Phase_N`, where N is the phase number.
3. Add a Colab secret named `HF` that holds the access token of the course data repository. The configuration cell reads it as `HF_TOKEN`, and the repository identifier as `HF_DATASET_REPO`. The first cells of every notebook download the data.
4. Download the shared checkpoints (see the top-level README) into `MyDrive`, keeping the sub-folder names, or run the Phase 1 training cells to regenerate them.
5. Run the notebook **from top to bottom**. Later sections reuse the tokenizer, model classes and helper functions defined earlier, and stop with an assertion if one of them is missing.
6. Each phase section ends by writing `prediction*.json` files in the format expected by the leaderboard.
