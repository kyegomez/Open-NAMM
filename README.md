[![Multi-Modality](agorabanner.png)](https://discord.com/servers/agora-999382051935506503)

# [Paper Implementation] AN EVOLVED UNIVERSAL TRANSFORMER MEMORY

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/agora-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)

An open source implementation of the paper: "AN EVOLVED UNIVERSAL TRANSFORMER MEMORY"


Abstract:

Prior methods propose to offset the escalating costs of modern foundation models by dropping specific parts of their contexts with hand-designed rules, while attempting to preserve their original performance. We overcome this trade-off with Neural Attention Memory Models (NAMMs), introducing a learned network for memory management that improves both the performance and efficiency of
transformers. We evolve NAMMs atop pre-trained transformers to provide different latent contexts focusing on the most relevant information for individual layers and attention heads. NAMMs are universally applicable to any model using selfattention as they condition exclusively on the values in the produced attention matrices. Learning NAMMs on a small set of problems, we achieve substantial performance improvements across multiple long-context benchmarks while cutting the model’s input contexts up to a fraction of the original sizes. We show the generality of our conditioning enables zero-shot transfer of NAMMs trained only on language to entirely new transformer architectures even across input modalities, with their benefits carrying over to vision and reinforcement learning.


## Usage

```python

def create_sample_inputs(
    batch_size: int = 2,
    seq_len: int = 1024,
    n_queries: int = 512,
    d_model: int = 256,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Create sample inputs for NAMM testing.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length (number of tokens in KV cache)
        n_queries: Number of recent queries
        d_model: Model dimension
        device: Device to create tensors on
    
    Returns:
        Tuple of (kv_cache, attention_matrix)
    """
    logger.info(f"Creating sample inputs on device: {device}")
    
    # Create sample KV cache
    # In practice, these would be the key and value tensors from transformer layers
    kv_cache = {
        "key": torch.randn(batch_size, seq_len, d_model, device=device),
        "value": torch.randn(batch_size, seq_len, d_model, device=device)
    }
    
    # Create sample attention matrix
    # In practice, this would be the recent attention scores from transformer layers
    attention_matrix = torch.randn(batch_size, seq_len, n_queries, device=device)
    
    # Apply softmax to make it look like real attention scores
    attention_matrix = torch.softmax(attention_matrix, dim=1)
    
    logger.info(
        f"Created inputs - KV cache size: {kv_cache['key'].shape}, "
        f"Attention matrix size: {attention_matrix.shape}"
    )
    
    return kv_cache, attention_matrix

def main():
    """Main function demonstrating NAMM usage."""
    # Setup logging
    logger.remove()
    logger.add(lambda msg: print(msg, flush=True), colorize=True, level="INFO")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create NAMM instance with custom config
    config = NAMMConfig(
        update_interval=256,  # More frequent updates for demonstration
        stride_size=16,
        window_size=64,
        d_model=256,
        n_head=4,
        gamma=0.95,
        dropout=0.1
    )
    
    namm = create_namm(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    namm = namm.to(device)
    
    logger.info(f"Created NAMM model on device: {device}")
    
    # Create sample inputs
    kv_cache, attention_matrix = create_sample_inputs(
        batch_size=2,
        seq_len=1024,
        n_queries=512,
        d_model=config.d_model,
        device=device
    )
    
    # Simulate multiple steps of processing
    n_steps = 1000
    retention_stats = []
    
    logger.info(f"Starting simulation for {n_steps} steps")
    
    for step in range(n_steps):
        # Process the KV cache
        updated_cache, _ = namm(kv_cache, attention_matrix)
        
        # Every few steps, evaluate retention
        if step % 100 == 0:
            stats = namm.evaluate_retention(kv_cache, attention_matrix)
            if stats:  # Only store if we got stats (remember NAMM only updates every update_interval)
                retention_stats.append(stats)
                logger.info(
                    f"Step {step}: Retention rate = {stats['retention_rate']:.2%}, "
                    f"Mean score = {stats['mean_score']:.3f}"
                )
        
        # Update KV cache and attention matrix for next step
        if updated_cache:  # If NAMM made updates
            kv_cache = updated_cache
            # Create new attention matrix for reduced sequence length
            _, new_seq_len, _ = kv_cache['key'].shape
            attention_matrix = torch.randn(
                2, new_seq_len, 512, device=device
            )
            attention_matrix = torch.softmax(attention_matrix, dim=1)
    
    # Print final statistics
    if retention_stats:
        avg_retention = sum(s['retention_rate'] for s in retention_stats) / len(retention_stats)
        logger.info(f"Average retention rate over simulation: {avg_retention:.2%}")

if __name__ == "__main__":
    main()
```


# License
MIT
