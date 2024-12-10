from dataclasses import dataclass
import math
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
            batch_first=True,
        )

        # Final linear layer for scoring
        self.score_proj = nn.Linear(config.d_model, 1)

        # Position embeddings
        self.pos_embedding = nn.Parameter(
            torch.randn(1, 1024, config.d_model)
        )

        logger.info(f"Initialized BAM network with config: {config}")

    def create_backward_mask(self, size: int) -> Tensor:
        """Create a backward (counter-causal) attention mask.

        Args:
            size: Size of the sequence

        Returns:
            Tensor: Boolean mask of shape (size, size)
        """
        mask = torch.ones(size, size, dtype=torch.bool)
        mask = torch.triu(
            mask, diagonal=1
        )  # Upper triangular without diagonal
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
            features,
            features,
            features,
            attn_mask=mask,
            need_weights=False,
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
        self.register_buffer("past_stft", None)
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
            device=attention_values.device,
        )

        # Compute STFT
        stft = torch.stft(
            attention_values.reshape(-1, n_queries),
            n_fft=self.config.window_size,
            hop_length=self.config.stride_size,
            window=window,
            return_complex=True,
        )

        # Get magnitude spectrum
        stft = torch.abs(stft)

        # Reshape to (batch_size, seq_len, time, freq)
        stft = stft.reshape(batch_size, seq_len, -1, stft.size(-2))

        # Project to d_model dimension
        if not hasattr(self, "feature_proj"):
            self.feature_proj = nn.Linear(
                stft.size(-1) * stft.size(-2),
                self.config.d_model,
                device=stft.device,
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
                    mode="linear",
                    align_corners=False,
                ).transpose(
                    1, 2
                )  # [B, S, D]
            else:
                past_features = self.past_stft

            reduced = reduced + (self.config.gamma * past_features)

        return reduced

    def forward(
        self,
        kv_cache: Dict[str, Tensor],
        attention_matrix: Tensor,
        *,
        return_scores: bool = False,
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
            kept_tokens_per_batch = keep_mask.sum(
                dim=1
            )  # [batch_size]
            max_tokens = kept_tokens_per_batch.max().item()

            # Initialize tensor for kept tokens
            new_tensor = torch.zeros(
                batch_size,
                max_tokens,
                d_model,
                device=v.device,
                dtype=v.dtype,
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
        self, kv_cache: Dict[str, Tensor], attention_matrix: Tensor
    ) -> Dict[str, float]:
        """Evaluate token retention statistics.

        Args:
            kv_cache: Current KV cache
            attention_matrix: Recent attention values

        Returns:
            Dict containing retention statistics
        """
        _, scores = self.forward(
            kv_cache, attention_matrix, return_scores=True
        )

        if scores is None:
            return {}

        keep_mask = scores > 0

        stats = {
            "retention_rate": keep_mask.float().mean().item(),
            "mean_score": scores.mean().item(),
            "score_std": scores.std().item(),
            "min_score": scores.min().item(),
            "max_score": scores.max().item(),
        }

        logger.debug(f"Retention statistics: {stats}")
        return stats


def create_namm(
    config: Optional[NAMMConfig] = None, **kwargs: Any
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


@dataclass
class TransformerConfig:
    """Configuration for Transformer with NAMM.

    Attributes:
        vocab_size: Size of the vocabulary
        max_seq_length: Maximum sequence length
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Dimension of feed-forward network
        dropout: Dropout rate
        activation: Activation function ("relu" or "gelu")
        use_namm: Whether to use NAMM for KV cache management
        namm_config: Configuration for NAMM module
    """

    vocab_size: int = 50257  # GPT-2 vocabulary size
    max_seq_length: int = 2048
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072
    dropout: float = 0.1
    activation: str = "gelu"
    use_namm: bool = True
    namm_config: Optional[NAMMConfig] = None


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional NAMM support."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0

        self.config = config
        self.d_head = config.d_model // config.n_heads
        self.n_heads = config.n_heads

        # Linear projections
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.o_proj = nn.Linear(config.d_model, config.d_model)

        self.dropout = nn.Dropout(config.dropout)

        # Initialize NAMM if enabled
        self.namm = None
        if config.use_namm:
            namm_config = config.namm_config or NAMMConfig(
                d_model=config.d_model, n_head=config.n_heads
            )
            self.namm = create_namm(namm_config)
            logger.info("Initialized NAMM for attention layer")

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        is_causal: bool = True,
    ) -> Tensor:
        batch_size, seq_len, _ = x.shape

        # Project queries, keys, and values
        q = self.q_proj(x).view(
            batch_size, seq_len, self.n_heads, self.d_head
        )
        k = self.k_proj(x).view(
            batch_size, seq_len, self.n_heads, self.d_head
        )
        v = self.v_proj(x).view(
            batch_size, seq_len, self.n_heads, self.d_head
        )

        # Transpose for attention
        q = q.transpose(1, 2)  # [B, H, S, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            self.d_head
        )

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        elif is_causal:
            causal_mask = torch.triu(
                torch.ones(
                    seq_len,
                    seq_len,
                    dtype=torch.bool,
                    device=x.device,
                ),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))

        # Compute attention weights and apply to values
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Update KV cache using NAMM if enabled
        if self.namm is not None and self.training:
            kv_cache = {"key": k, "value": v}
            # Get mean attention across heads for NAMM
            mean_attn = attn.mean(dim=1)  # [B, S, S]
            kv_cache, _ = self.namm(kv_cache, mean_attn)
            k = kv_cache["key"]
            v = kv_cache["value"]

        out = torch.matmul(attn, v)

        # Reshape and project output
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, -1)
        )
        out = self.o_proj(out)

        return out


class TransformerBlock(nn.Module):
    """Transformer block with NAMM support."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU() if config.activation == "gelu" else nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        # Self-attention with residual
        attn = self.attention(self.norm1(x), mask)
        x = x + attn

        # Feed-forward with residual
        x = x + self.ff(self.norm2(x))
        return x


class NAMMTransformer(nn.Module):
    """Transformer model with NAMM support."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.d_model
        )
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, config.max_seq_length, config.d_model)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        # Final layer norm
        self.norm = nn.LayerNorm(config.d_model)

        # Output projection
        self.output = nn.Linear(config.d_model, config.vocab_size)

        logger.info(
            f"Initialized NAMMTransformer with config: {config}"
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size, seq_len = input_ids.shape

        # Get embeddings
        x = self.token_embedding(input_ids)
        x = x + self.pos_embedding[:, :seq_len]

        # Process through transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)

        # Final norm and output projection
        x = self.norm(x)
        logits = self.output(x)

        return logits


def create_namm_transformer(
    config: Optional[TransformerConfig] = None, **kwargs: Any
) -> NAMMTransformer:
    """Create a NAMMTransformer instance with given config.

    Args:
        config: Configuration object, if None uses default config
        **kwargs: Override default config values

    Returns:
        NAMMTransformer: Initialized transformer instance
    """
    if config is None:
        config = TransformerConfig()

    # Update config with any provided kwargs
    for k, v in kwargs.items():
        if hasattr(config, k):
            setattr(config, k, v)

    return NAMMTransformer(config)
