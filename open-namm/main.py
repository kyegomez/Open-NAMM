from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch import Tensor


@dataclass
class NAMMConfig:
    """Configuration for Neural Attention Memory Model.
    
    Attributes:
        update_interval: Number of steps between NAMM updates (n_up)
        stride_size: Size of the stride for STFT computation (s_w)
        window_size: Size of the Hann window for STFT
        n_head: Number of attention heads in the BAM network
        d_model: Dimension of the feature vectors
        gamma: Decay factor for exponential moving average
        dropout: Dropout rate for the BAM network
    """
    update_interval: int = 512
    stride_size: int = 32
    window_size: int = 128
    n_head: int = 4
    d_model: int = 256
    gamma: float = 0.95
    dropout: float = 0.1

class BackwardAttentionMemory(nn.Module):
    """Backward Attention Memory (BAM) network for token importance scoring."""
    
    def __init__(self, config: NAMMConfig):
        """Initialize the BAM network.
        
        Args:
            config: Configuration object containing model parameters
        """
        super().__init__()
        self.config = config
        
        # Multi-head attention with backward masking
        self.self_attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_head,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Final linear layer for scoring
        self.score_proj = nn.Linear(config.d_model, 1)
        
        # Position embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, 1024, config.d_model))
        
        logger.info(f"Initialized BAM network with config: {config}")

    def create_backward_mask(self, size: int) -> Tensor:
        """Create a backward (counter-causal) attention mask.
        
        Args:
            size: Size of the sequence
            
        Returns:
            Tensor: Boolean mask of shape (size, size)
        """
        mask = torch.ones(size, size, dtype=torch.bool)
        mask = torch.triu(mask, diagonal=1)  # Upper triangular without diagonal
        return mask

    def forward(self, features: Tensor) -> Tensor:
        """Process features through the BAM network.
        
        Args:
            features: Token features of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor: Importance scores for each token
        """
        batch_size, seq_len = features.shape[:2]
        
        # Add positional embeddings with proper size
        pos_emb = self.pos_embedding[:, :seq_len, :]
        features = features + pos_emb
        
        # Create backward mask
        mask = self.create_backward_mask(seq_len).to(features.device)
        
        # Apply self-attention with backward masking
        attended, _ = self.self_attention(
            features, features, features,
            attn_mask=mask,
            need_weights=False
        )
        
        # Generate scores
        scores = self.score_proj(attended).squeeze(-1)
        
        return scores

class NAMM(nn.Module):
    """Neural Attention Memory Model for efficient KV cache management."""
    
    def __init__(self, config: NAMMConfig):
        """Initialize the NAMM.
        
        Args:
            config: Configuration object containing model parameters
        """
        super().__init__()
        self.config = config
        self.bam = BackwardAttentionMemory(config)
        self.register_buffer('past_stft', None)
        self._step = 0
        
        logger.info("Initialized NAMM")
    
    def compute_stft(self, attention_values: Tensor) -> Tensor:
        """Compute Short-Time Fourier Transform of attention values.
        
        Args:
            attention_values: Attention values of shape (batch_size, seq_len, n_queries)
            
        Returns:
            Tensor: STFT features
        """
        batch_size, seq_len, n_queries = attention_values.shape
        
        # Create Hann window
        window = torch.hann_window(
            self.config.window_size,
            periodic=True,
            device=attention_values.device
        )
        
        # Compute STFT
        stft = torch.stft(
            attention_values.reshape(-1, n_queries),
            n_fft=self.config.window_size,
            hop_length=self.config.stride_size,
            window=window,
            return_complex=True
        )
        
        # Get magnitude spectrum
        stft = torch.abs(stft)
        
        # Reshape to (batch_size, seq_len, time, freq)
        stft = stft.reshape(batch_size, seq_len, -1, stft.size(-2))
        
        # Project to d_model dimension
        if not hasattr(self, 'feature_proj'):
            self.feature_proj = nn.Linear(
                stft.size(-1) * stft.size(-2),
                self.config.d_model,
                device=stft.device
            )
        
        # Flatten last two dimensions and project
        stft_flat = stft.reshape(batch_size, seq_len, -1)
        stft = self.feature_proj(stft_flat)
        
        return stft

    def reduce_features(self, stft_features: Tensor) -> Tensor:
        """Reduce STFT features using exponential moving average.
        
        Args:
            stft_features: STFT features of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor: Reduced features
        """
        reduced = stft_features
        
        # Add past STFT if available
        if self.past_stft is not None:
            # Handle different sequence lengths
            if self.past_stft.size(1) != reduced.size(1):
                # Interpolate past_stft to match current sequence length
                past_features = F.interpolate(
                    self.past_stft.transpose(1, 2),  # [B, D, S]
                    size=reduced.size(1),
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)  # [B, S, D]
            else:
                past_features = self.past_stft
                
            reduced = reduced + (self.config.gamma * past_features)
                
        return reduced

   
    def forward(
        self,
        kv_cache: Dict[str, Tensor],
        attention_matrix: Tensor,
        *,
        return_scores: bool = False
    ) -> Tuple[Dict[str, Tensor], Optional[Tensor]]:
        """Process KV cache using NAMM.
        
        Args:
            kv_cache: Dictionary containing key and value tensors
            attention_matrix: Recent attention values of shape (batch_size, seq_len, n_queries)
            return_scores: Whether to return importance scores
            
        Returns:
            Tuple containing:
                - Updated KV cache
                - Optional tensor of importance scores if return_scores is True
        """
        self._step += 1
        
        # Only process every update_interval steps
        if self._step % self.config.update_interval != 0:
            return kv_cache, None
            
        logger.debug(f"Processing NAMM at step {self._step}")
        
        # Compute STFT features
        stft_features = self.compute_stft(attention_matrix)
        
        # Reduce features with EMA
        reduced_features = self.reduce_features(stft_features)
        
        # Update past STFT
        self.past_stft = reduced_features.detach()
        
        # Get importance scores from BAM
        scores = self.bam(reduced_features)
        
        # Create mask for tokens to keep
        keep_mask = scores > 0  # Shape: [batch_size, seq_len]
        
        # Update KV cache with proper handling of batch dimension
        updated_cache = {}
        for k, v in kv_cache.items():
            # Handle each batch separately
            batch_size = v.size(0)
            d_model = v.size(-1)
            kept_tokens_per_batch = keep_mask.sum(dim=1)  # [batch_size]
            max_tokens = kept_tokens_per_batch.max().item()
            
            # Initialize tensor for kept tokens
            new_tensor = torch.zeros(
                batch_size, max_tokens, d_model,
                device=v.device, dtype=v.dtype
            )
            
            # Process each batch element
            for b in range(batch_size):
                # Get indices of tokens to keep for this batch
                keep_indices = keep_mask[b].nonzero().squeeze(-1)
                n_tokens = keep_indices.size(0)
                
                # Select and store kept tokens
                new_tensor[b, :n_tokens] = v[b, keep_indices]
            
            updated_cache[k] = new_tensor
        
        logger.info(
            f"NAMM update complete. "
            f"Retained {keep_mask.float().mean():.2%} of tokens "
            f"(max {max_tokens} tokens per batch)"
        )
        
        return updated_cache, scores if return_scores else None
   
    @torch.no_grad()
    def evaluate_retention(
        self,
        kv_cache: Dict[str, Tensor],
        attention_matrix: Tensor
    ) -> Dict[str, float]:
        """Evaluate token retention statistics.
        
        Args:
            kv_cache: Current KV cache
            attention_matrix: Recent attention values
            
        Returns:
            Dict containing retention statistics
        """
        _, scores = self.forward(
            kv_cache,
            attention_matrix,
            return_scores=True
        )
        
        if scores is None:
            return {}
            
        keep_mask = scores > 0
        
        stats = {
            "retention_rate": keep_mask.float().mean().item(),
            "mean_score": scores.mean().item(),
            "score_std": scores.std().item(),
            "min_score": scores.min().item(),
            "max_score": scores.max().item()
        }
        
        logger.debug(f"Retention statistics: {stats}")
        return stats

def create_namm(
    config: Optional[NAMMConfig] = None,
    **kwargs: Any
) -> NAMM:
    """Create a NAMM instance with given config.
    
    Args:
        config: Configuration object, if None uses default config
        **kwargs: Override default config values
        
    Returns:
        NAMM: Initialized NAMM instance
    """
    if config is None:
        config = NAMMConfig()
        
    # Update config with any provided kwargs
    for k, v in kwargs.items():
        if hasattr(config, k):
            setattr(config, k, v)
    
    return NAMM(config)


# import torch
# from loguru import logger

# Import from previous implementation
# from namm import create_namm, NAMMConfig

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