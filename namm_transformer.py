# Add transformer demo to main function
from loguru import logger
import torch

from open_namm.main import (
    NAMMConfig,
    TransformerConfig,
    create_namm,
    create_namm_transformer,
)


def main():
    """Main function demonstrating NAMM and Transformer usage."""
    # [Previous main function code remains the same until the end]
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

    create_namm(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Add transformer demo
    logger.info("Starting transformer demo...")

    # Create transformer config
    transformer_config = TransformerConfig(
        vocab_size=1000,  # Smaller vocab for demo
        max_seq_length=128,
        d_model=256,
        n_heads=4,
        n_layers=2,
        use_namm=True,
        namm_config=config,  # Reuse NAMM config from above
    )

    # Create transformer
    transformer = create_namm_transformer(transformer_config)
    transformer = transformer.to(device)

    # Create sample input
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(
        0,
        transformer_config.vocab_size,
        (batch_size, seq_len),
        device=device,
    )

    # Forward pass
    logits = transformer(input_ids)

    logger.info(
        f"Transformer forward pass successful. "
        f"Output shape: {logits.shape}"
    )


if __name__ == "__main__":
    main()
