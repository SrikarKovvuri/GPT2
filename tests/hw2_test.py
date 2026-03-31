"""
Test file to verify GPT-2 implementation against HuggingFace's implementation.

This file imports transformers (HF) to compare outputs between the custom
implementation and the reference implementation.

Usage:
    # Make sure you are at the root of the hw2 directory.
    # Run all tests:
    pytest hw2_test.py -v
    # Run a specific test:
    pytest hw2_test.py::test_loads_and_forward_pass -v
"""

import json
import pytest
import torch
import torch.nn.functional as F

# Import HuggingFace transformers for comparison
from transformers import GPT2LMHeadModel as HFGPT2LMHeadModel, GPT2Tokenizer

# Import our custom implementation
from src.gpt2 import GPT2Config, GPT2LMHeadModel, GPT2ForSequenceClassification


@pytest.fixture(scope="module")
def device():
    """Get the device to run tests on."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def num_samples(pytestconfig):
    """Get the number of samples to test."""
    return pytestconfig.getoption("--num-samples")


@pytest.fixture(scope="module")
def max_length(pytestconfig):
    """Get the maximum sequence length."""
    return pytestconfig.getoption("--max-length")


@pytest.fixture(scope="module")
def gpt2_bin_path(pytestconfig):
    """Get the path to the local PyTorch model bin file."""
    return pytestconfig.getoption("--gpt2-bin-path")


@pytest.fixture(scope="module")
def openwebtext_data_path(pytestconfig):
    """Get the path to the OpenWebText tokenized data file."""
    return pytestconfig.getoption("--openwebtext-data-path")


@pytest.fixture(scope="module")
def hf_model_name(pytestconfig):
    """Get the HuggingFace model name."""
    return pytestconfig.getoption("--hf-model")


@pytest.fixture(scope="module")
def batch_size(pytestconfig):
    """Get the batch size for testing."""
    return pytestconfig.getoption("--batch-size")


@pytest.fixture(scope="module")
def prefill_tokens(pytestconfig):
    """Get the number of prefill tokens for generation tests."""
    return pytestconfig.getoption("--prefill-tokens")


@pytest.fixture(scope="module")
def gen_tokens(pytestconfig):
    """Get the number of tokens to generate in generation tests."""
    return pytestconfig.getoption("--gen-tokens")


@pytest.fixture(scope="module")
def temperature(pytestconfig):
    """Get the temperature for nucleus sampling test."""
    return pytestconfig.getoption("--temperature")


@pytest.fixture(scope="module")
def top_p(pytestconfig):
    """Get the top-p threshold for nucleus sampling test."""
    return pytestconfig.getoption("--top-p")


@pytest.fixture(scope="module")
def classifier_bin_path(pytestconfig):
    """Get the path to the classifier model bin file."""
    return pytestconfig.getoption("--classifier-bin-path")


@pytest.fixture(scope="module")
def classifier_test_file(pytestconfig):
    """Get the path to the classifier test data file."""
    return pytestconfig.getoption("--classifier-test-file")


@pytest.fixture(scope="module")
def classifier_accuracy_threshold(pytestconfig):
    """Get the accuracy threshold for classifier test."""
    return pytestconfig.getoption("--classifier-accuracy-threshold")


@pytest.fixture(scope="module")
def tokenizer(hf_model_name):
    """Load GPT2 tokenizer for converting tokens to strings."""
    return GPT2Tokenizer.from_pretrained(hf_model_name)


@pytest.fixture(scope="module")
def hf_model(device, hf_model_name):
    """Load HuggingFace pretrained model."""
    model = HFGPT2LMHeadModel.from_pretrained(hf_model_name)
    model.to(device)
    model.eval()
    return model


@pytest.fixture(scope="module")
def custom_model(device, gpt2_bin_path):
    """Load custom GPT-2 model from bin file."""
    import os
    config = GPT2Config()
    model = GPT2LMHeadModel(config=config, bin_path=gpt2_bin_path)
    model.to(device)
    model.eval()
    return model


@pytest.fixture(scope="module")
def sample_data(num_samples, max_length, openwebtext_data_path):
    """Load sample tokenized data from jsonl file and pad to max_length.
    
    Returns:
        List of padded token sequences, all with length=max_length.
        Padding uses token_id=0 (eos_token_id for GPT-2).
    """
    jsonl_path = openwebtext_data_path
    samples = []
    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            data = json.loads(line)
            tokens = data["token_ids"]
            # Truncate to max_length
            truncated = tokens[:max_length]
            # Pad to max_length with 0 (eos_token_id for GPT-2)
            padded = truncated + [0] * (max_length - len(truncated))
            samples.append(padded)
    return samples


@pytest.fixture(scope="module")
def classifier_test_data(classifier_test_file, max_length):
    """Load classifier test data from jsonl file and pad to max_length.
    
    Returns:
        List of dicts with 'token_ids' (padded to max_length) and 'label'.
        Padding uses token_id=0 (eos_token_id for GPT-2).
    """
    samples = []
    with open(classifier_test_file, "r") as f:
        for line in f:
            data = json.loads(line)
            token_ids = data["token_ids"]
            label = data["label"]
            # Truncate if too long
            truncated = token_ids[:max_length]
            # Pad to max_length with 0 (eos_token_id for GPT-2)
            padded = truncated + [0] * (max_length - len(truncated))
            samples.append({"token_ids": padded, "label": label})
    return samples


def get_probability_distributions(hf_model, custom_model, sample_tokens_batch, device):
    """Get probability distributions from both models for a batch of inputs.
    
    Args:
        hf_model: HuggingFace GPT-2 model
        custom_model: Custom GPT-2 model
        sample_tokens_batch: List of token sequences (batch), already padded
        device: Device to run on
    
    Returns:
        Tuple of (hf_probs, custom_probs) tensors with shape [batch_size, seq_len, vocab_size]
    """
    # Data is already padded by sample_data fixture, just convert to tensor
    input_ids = torch.tensor(sample_tokens_batch, dtype=torch.long, device=device)
    
    with torch.no_grad():
        hf_logits = hf_model(input_ids).logits
        custom_logits = custom_model(input_ids).logits
        
        # Get probabilities
        hf_probs = F.softmax(hf_logits, dim=-1)
        custom_probs = F.softmax(custom_logits, dim=-1)
    
    return hf_probs, custom_probs


def batch_generator(data, batch_size):
    """Generate batches from a list."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def test_loads_and_forward_pass(device, gpt2_bin_path, sample_data, batch_size, max_length):
    """
    Test that the student GPT2LMHeadModel loads correctly and performs forward pass.
    
    This test verifies:
    1. The model can be instantiated and loaded from bin_path
    2. The model can perform a forward pass without errors on all loaded data
    3. The output shape is correct (batch_size, seq_len, vocab_size) for each batch
    
    Only checks output shape, not the actual values.
    """
    # Load the student model
    config = GPT2Config()
    model = GPT2LMHeadModel(config=config, bin_path=gpt2_bin_path)
    model.to(device)
    model.eval()
    
    # Process all loaded data in batches
    for batch_idx, batch in enumerate(batch_generator(sample_data, batch_size)):
        # Data is already padded by sample_data fixture, just convert to tensor
        input_ids = torch.tensor(batch, dtype=torch.long, device=device)
        
        # Perform forward pass
        with torch.no_grad():
            output = model(input_ids)
        
        # Check that output has the expected attributes
        assert hasattr(output, "logits"), "Model output should have 'logits' attribute"
        
        # Check output shape: [batch_size, seq_len, vocab_size]
        # GPT-2 small has vocab_size=50257
        expected_shape = (len(batch), max_length, 50257)
        actual_shape = output.logits.shape
        
        assert actual_shape == expected_shape, (
            f"Batch {batch_idx+1}: Output shape mismatch. Expected {expected_shape}, got {actual_shape}"
        )


def test_probability_tolerance(hf_model, custom_model, sample_data, device, batch_size):
    """
    Test that probability distributions match within tolerance.
    
    Checks that the maximum probability difference is within 1e-4.
    Processes samples in batches for efficiency.
    """
    tolerance = 1e-4
    
    for batch_idx, batch in enumerate(batch_generator(sample_data, batch_size)):
        hf_probs, custom_probs = get_probability_distributions(
            hf_model, custom_model, batch, device
        )
        
        # Compare probabilities for all positions (data is already padded to same length)
        diff = torch.abs(hf_probs - custom_probs)
        max_diff = torch.max(diff).item()
        
        assert max_diff < tolerance, (
            f"Batch {batch_idx+1}: Maximum probability difference {max_diff:.8e} "
            f"exceeds tolerance {tolerance:.8e}"
        )


def test_greedy_sampling(hf_model, custom_model, sample_data, device, tokenizer, batch_size, prefill_tokens, gen_tokens):
    """
    Test that batched generation with greedy sampling (temperature=0.0) matches
    between student implementation and HuggingFace reference.
    
    This test:
    1. Asserts all samples have at least `prefill_tokens` tokens
    2. Processes all samples in batches
    3. Uses first `prefill_tokens` tokens as prompt
    4. Generates `gen_tokens` tokens using greedy sampling (temperature=0.0)
    5. Compares generated token sequences between implementations
    6. Displays the generated text for debugging purposes
    """
    # Assert all samples have at least prefill_tokens
    # (sample_data is already padded to max_length, so we check original content)
    for i, tokens in enumerate(sample_data):
        # Find actual length (before padding with 0s)
        actual_len = len([t for t in tokens if t != 0]) if 0 in tokens else len(tokens)
        assert actual_len >= prefill_tokens, (
            f"Sample {i+1} has {actual_len} tokens, but needs at least {prefill_tokens}"
        )
    
    for batch_idx, batch in enumerate(batch_generator(sample_data, batch_size)):
        # Prepare batch: take first prefill_tokens from each sample
        prompt_batch = [tokens[:prefill_tokens] for tokens in batch]
        
        input_ids = torch.tensor(prompt_batch, dtype=torch.long, device=device)
        
        # Generate using custom implementation with greedy sampling
        with torch.no_grad():
            custom_output = custom_model.generate(
                input_ids.clone(),
                temperature=0.0,
                top_p=1.0,
                max_new_tokens=gen_tokens
            )
        
        # Generate using HF implementation with greedy sampling
        with torch.no_grad():
            hf_output = hf_model.generate(
                input_ids.clone(),
                max_new_tokens=gen_tokens,
                do_sample=False,  # Greedy sampling
                use_cache=True,
                eos_token_id=None,
            )
        
        # Compare generated sequences for each sample in the batch
        for i, sample in enumerate(batch):
            sample_idx = batch_idx * batch_size + i
            
            # Get generated sequences
            custom_seq = custom_output.sequences[i]
            hf_seq = hf_output[i]
            
            # Check shapes match
            assert custom_seq.shape == hf_seq.shape, (
                f"Sample {sample_idx+1}: Generated sequence shapes do not match. "
                f"Custom: {custom_seq.shape}, HF: {hf_seq.shape}"
            )
            
            # Check token sequences match exactly
            assert torch.equal(custom_seq, hf_seq), (
                f"Sample {sample_idx+1}: Generated token sequences do not match.\n"
                f"  Custom: {custom_seq.tolist()}\n"
                f"  HF:     {hf_seq.tolist()}"
            )
            
            # Convert to strings for display/debugging (only for first few samples)
            if sample_idx < 3:
                prompt_text = tokenizer.decode(prompt_batch[i], skip_special_tokens=True)
                custom_generated_text = tokenizer.decode(custom_seq, skip_special_tokens=True)
                hf_generated_text = tokenizer.decode(hf_seq, skip_special_tokens=True)
                
                print(f"\n--- Sample {sample_idx+1} ---")
                print(f"Prompt ({len(prompt_batch[i])} tokens): {prompt_text!r}")
                print(f"Custom generated ({custom_seq.shape[0]} tokens): {custom_generated_text!r}")
                print(f"HF generated ({hf_seq.shape[0]} tokens): {hf_generated_text!r}")
                print(f"Match: {custom_generated_text == hf_generated_text}")


def test_nucleus_sampling(
    hf_model, custom_model, sample_data, device, batch_size,
    prefill_tokens, gen_tokens, temperature, top_p
):
    """
    Test that nucleus sampling generates plausible tokens for all samples.
    
    This test:
    1. Asserts all samples have at least `prefill_tokens` tokens
    2. Processes all samples in batches
    3. Uses first `prefill_tokens` tokens as prompt
    4. Generates `gen_tokens` tokens using nucleus sampling (temperature, top_p)
    5. Makes ONE HF forward pass with the full sequences (prefill + generated)
    6. For each position, computes HF logits with temperature scaling
    7. Applies top-p filtering to get the nucleus set and verifies the generated token
    
    This ensures the student's nucleus sampling implementation actually restricts
    sampling to the top-p cumulative probability mass.
    """
    # Assert all samples have at least prefill_tokens
    # (sample_data is already padded to max_length, so we check original content)
    for i, tokens in enumerate(sample_data):
        # Find actual length (before padding with 0s)
        actual_len = len([t for t in tokens if t != 0]) if 0 in tokens else len(tokens)
        assert actual_len >= prefill_tokens, (
            f"Sample {i+1} has {actual_len} tokens, but needs at least {prefill_tokens}"
        )
    
    for batch_idx, batch in enumerate(batch_generator(sample_data, batch_size)):
        # Prepare batch: take first prefill_tokens from each sample
        prompt_batch = [tokens[:prefill_tokens] for tokens in batch]
        
        input_ids = torch.tensor(prompt_batch, dtype=torch.long, device=device)
        
        # Generate using custom implementation with nucleus sampling
        with torch.no_grad():
            custom_output = custom_model.generate(
                input_ids.clone(),
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=gen_tokens
            )
        
        # Get generated sequences for this batch
        generated_sequences = custom_output.sequences  # [batch_size, seq_len]
        batch_size_actual = generated_sequences.shape[0]
        
        # Make ONE HF forward pass with the full sequences
        with torch.no_grad():
            hf_logits = hf_model(generated_sequences).logits  # [batch_size, total_len, vocab_size]
        
        # Verify each generated token position
        # We check positions prefill_tokens to total_len-1 (the generated tokens)
        for step in range(gen_tokens):
            position = prefill_tokens + step  # Position in the sequence (0-indexed)
            
            # Get logits for this position from all samples
            # hf_logits[:, position-1, :] gives logits for predicting position-th token
            # (since logits are shifted by 1 - logits at position i predict token i+1)
            next_token_logits = hf_logits[:, position - 1, :]  # [batch_size, vocab_size]
            
            # Apply temperature scaling
            scaled_logits = next_token_logits / temperature
            
            # Compute probabilities for all samples at once
            probs = F.softmax(scaled_logits, dim=-1)  # [batch_size, vocab_size]
            
            # Apply top-p filtering logic to find the nucleus set for each sample
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Find tokens in the nucleus for each sample (cumulative prob <= top_p)
            nucleus_mask = cumulative_probs <= top_p
            # Always include the top token for each sample
            nucleus_mask[:, 0] = True
            
            # For each sample in the batch, verify the generated token is in its nucleus
            for i in range(batch_size_actual):
                sample_idx = batch_idx * batch_size + i
                gen_token = generated_sequences[i, position].item()
                
                # Get the nucleus tokens for this sample
                sample_nucleus_mask = nucleus_mask[i]
                sample_sorted_indices = sorted_indices[i]
                nucleus_tokens = sample_sorted_indices[sample_nucleus_mask].tolist()
                
                # Verify the generated token is in the nucleus set
                assert gen_token in nucleus_tokens, (
                    f"Sample {sample_idx+1}, position {position}: "
                    f"Generated token {gen_token} is not in the nucleus set. "
                    f"Nucleus contains {len(nucleus_tokens)} tokens with cumulative prob <= {top_p}. "
                    f"Token probability: {probs[i, gen_token].item():.6f}"
                )


def test_classifier_loads_and_forward_pass(device, batch_size, gpt2_bin_path, classifier_test_data, max_length):
    """
    Test that GPT2ForSequenceClassification loads correctly and performs forward pass.
    
    This test verifies:
    1. The classifier can be instantiated with 20 labels
    2. The classifier loads GPT-2 base LM weights from the pytorch checkpoint by specifying the bin path
    3. The classifier can perform a forward pass without errors on real data
    4. The output logits have shape (batch_size, num_labels) = (batch_size, 20)
    
    Always uses batching. Data is truncated to max_length (1024) and padded by the fixture.
    """
    num_labels = 20
    # Initialize the classifier model with 20 labels and load weights from gpt2_bin_path
    config = GPT2Config()
    config.num_labels = num_labels
    model = GPT2ForSequenceClassification(
        config=config,
        lm_bin_path=gpt2_bin_path
    )
    model.to(device)
    model.eval()
    
    # Process all test data in batches
    for batch_idx, batch in enumerate(batch_generator(classifier_test_data, batch_size)):
        # Data is already padded by classifier_test_data fixture, just convert to tensor
        input_ids = torch.tensor([s["token_ids"] for s in batch], dtype=torch.long, device=device)
        
        # Perform forward pass
        with torch.no_grad():
            output = model(input_ids)
        
        # Check that output has the expected attributes
        assert hasattr(output, "logits"), "Model output should have 'logits' attribute"
        
        # Check output shape: [batch_size, num_labels]
        expected_shape = (len(batch), num_labels)
        actual_shape = output.logits.shape
        
        assert actual_shape == expected_shape, (
            f"Batch {batch_idx+1}: Output shape mismatch. Expected {expected_shape}, got {actual_shape}"
        )
        
        # Verify the number of labels is 20
        assert output.logits.shape[-1] == 20, (
            f"Number of labels should be 20, got {output.logits.shape[-1]}"
        )


def test_classifier_accuracy(device, batch_size, classifier_bin_path, classifier_test_data, 
                             classifier_accuracy_threshold):
    """
    Test that the loaded classifier achieves an accuracy threshold on a test file.
    
    This test:
    1. Loads the classifier from the specified bin path
    2. Evaluates on the test file
    3. Checks if accuracy meets or exceeds the threshold
    
    Always uses batching for processing. Test will fail if files do not exist.
    
    Args:
        device: Device to run on
        batch_size: Batch size for evaluation
        classifier_bin_path: Path to the classifier model checkpoint
        classifier_test_data: Loaded test data samples (already padded)
        classifier_accuracy_threshold: Minimum accuracy threshold
    """
    num_labels = 20
    
    # Load the classifier model
    config = GPT2Config()
    config.num_labels = num_labels
    model = GPT2ForSequenceClassification(
        config=config,
        classifier_bin_path=classifier_bin_path
    )
    model.to(device)
    model.eval()
    
    # Evaluate with batching
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_start in range(0, len(classifier_test_data), batch_size):
            batch_end = min(batch_start + batch_size, len(classifier_test_data))
            batch = classifier_test_data[batch_start:batch_end]
            
            # Data is already padded by classifier_test_data fixture, just convert to tensors
            input_ids = torch.tensor([s["token_ids"] for s in batch], dtype=torch.long, device=device)
            labels = torch.tensor([s["label"] for s in batch], dtype=torch.long, device=device)
            
            # Forward pass
            outputs = model(input_ids)
            logits = outputs.logits
            
            # Get predictions
            predictions = torch.argmax(logits, dim=-1)
            
            # Count correct predictions
            correct = (predictions == labels).sum().item()
            total_correct += correct
            total_samples += len(batch)
    
    # Calculate accuracy
    accuracy = total_correct / total_samples
    
    # Assert accuracy meets threshold
    assert accuracy >= classifier_accuracy_threshold, (
        f"Classifier accuracy {accuracy:.4f} ({accuracy*100:.2f}%) is below "
        f"threshold {classifier_accuracy_threshold:.4f} ({classifier_accuracy_threshold*100:.2f}%). "
        f"Correct: {total_correct}/{total_samples}"
    )
    
    print(f"\nClassifier accuracy: {accuracy:.4f} ({accuracy*100:.2f}%) - "
          f"Threshold: {classifier_accuracy_threshold:.4f} ({classifier_accuracy_threshold*100:.2f}%)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
