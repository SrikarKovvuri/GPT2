"""
Pytest configuration file for GPT-2 tests.

This file defines custom command-line options for test configuration.
"""


def pytest_addoption(parser):
    """Add custom command-line options for test configuration."""
    parser.addoption(
        "--num-samples",
        type=int,
        default=256,
        help="Number of samples to test (default: 256)",
    )
    parser.addoption(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length for each sample (default: 512)",
    )
    parser.addoption(
        "--gpt2-bin-path",
        type=str,
        default="./checkpoints/gpt2_model.pth",
        help="Path to the local PyTorch model bin file (default: ./checkpoints/gpt2_model.pth)",
    )
    parser.addoption(
        "--openwebtext-data-path",
        type=str,
        default="./data/openwebtext_1k_tokenized.jsonl",
        help="Path to the OpenWebText tokenized data file (default: ./data/openwebtext_1k_tokenized.jsonl)",
    )
    parser.addoption(
        "--hf-model",
        type=str,
        default="openai-community/gpt2",
        help="HuggingFace model name to compare against (default: openai-community/gpt2)",
    )
    parser.addoption(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for testing (default: 16)",
    )
    parser.addoption(
        "--prefill-tokens",
        type=int,
        default=128,
        help="Number of tokens to use as prefill/prompt for generation tests (default: 128)",
    )
    parser.addoption(
        "--gen-tokens",
        type=int,
        default=128,
        help="Number of tokens to generate in generation tests (default: 128)",
    )
    parser.addoption(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for nucleus sampling test (default: 0.7)",
    )
    parser.addoption(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) threshold for sampling test (default: 0.95)",
    )
    parser.addoption(
        "--classifier-bin-path",
        type=str,
        default="./checkpoints/classifier_model.pth",
        help="Path to the classifier model bin file (default: ./checkpoints/classifier_model.pth)",
    )
    parser.addoption(
        "--classifier-test-file",
        type=str,
        default="./data/20_newsgroups_val.jsonl",
        help="Path to the classifier test data file (default: ./data/20_newsgroups_val.jsonl)",
    )
    parser.addoption(
        "--classifier-accuracy-threshold",
        type=float,
        default=0.65,
        help="Accuracy threshold for classifier test (default: 0.65)",
    )
