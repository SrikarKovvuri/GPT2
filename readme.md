

## Overview
.

The project covers the following:

- Implementing a decoder-only transformer architecture from scratch using only Python and PyTorch, including:
    - Token and positional embeddings
    - Decoder-only Transformer blocks, including causal multi-head self-attention and MLP layers
    - A language modeling head for next-token prediction
- Implementing an auto-regressive generation function that supports:
    - Nucleus sampling with configurable temperature and top-p
    - Key-value caching to speed up generation
    - Batched generation to produce multiple sequences in parallel
- Verifying that the implementation can load the official PyTorch GPT-2 model checkpoint and is numerically consistent with the official GPT-2 implementation


For the GPT-2-based text classification model:

- Add a classification head on top of the GPT-2 language model
- Implement a training loop to fine-tune the GPT-2 classification model on the provided topic classification dataset
- Evaluate the fine-tuned model on a held-out validation set and report the classification accuracy