from loguru import logger
import torch
from open_namm.main import NAMMConfig, create_namm


def create_sample_inputs(
    batch_size: int = 2,
    seq_len: int = 1024,
    n_queries: int = 512,
    d_model: int = 256,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
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
        "key": torch.randn(
            batch_size, seq_len, d_model, device=device
        ),
        "value": torch.randn(
            batch_size, seq_len, d_model, device=device
        ),
    }

    # Create sample attention matrix
    # In practice, this would be the recent attention scores from transformer layers
    attention_matrix = torch.randn(
        batch_size, seq_len, n_queries, device=device
    )

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
    logger.add(
        lambda msg: print(msg, flush=True),
        colorize=True,
        level="INFO",
    )

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
        dropout=0.1,
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
        device=device,
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
            stats = namm.evaluate_retention(
                kv_cache, attention_matrix
            )
            if (
                stats
            ):  # Only store if we got stats (remember NAMM only updates every update_interval)
                retention_stats.append(stats)
                logger.info(
                    f"Step {step}: Retention rate = {stats['retention_rate']:.2%}, "
                    f"Mean score = {stats['mean_score']:.3f}"
                )

        # Update KV cache and attention matrix for next step
        if updated_cache:  # If NAMM made updates
            kv_cache = updated_cache
            # Create new attention matrix for reduced sequence length
            _, new_seq_len, _ = kv_cache["key"].shape
            attention_matrix = torch.randn(
                2, new_seq_len, 512, device=device
            )
            attention_matrix = torch.softmax(attention_matrix, dim=1)

    # Print final statistics
    if retention_stats:
        avg_retention = sum(
            s["retention_rate"] for s in retention_stats
        ) / len(retention_stats)
        logger.info(
            f"Average retention rate over simulation: {avg_retention:.2%}"
        )


if __name__ == "__main__":
    main()
