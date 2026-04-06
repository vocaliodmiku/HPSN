"""
Unit tests for ChunkedRecurrentVerticalAttention.

Tests:
  1. No cache → matches standard bottom-up behavior (only BU sources)
  2. With cache → top-down sources are included, output shape correct
  3. Per-token variant produces (B, T, N) attention weights
  4. Gradients don't flow through cache (detached)
  5. Separate vs shared top-down keys both work
"""

import torch
import torch.nn as nn
import pytest

from modules import (
    ContextConditionedVerticalAttention,
    ChunkedRecurrentVerticalAttention,
    SubspaceInhibition,
)


B, T, D, D_Q = 2, 10, 64, 16  # small dims for fast tests
NUM_BLOCKS = 5


def _make_block_outputs(num_blocks, B=B, T=T, D=D):
    """Create fake block outputs [o_0, o_1, ..., o_{num_blocks}]."""
    return [torch.randn(B, T, D) for _ in range(num_blocks + 1)]


class TestChunkedRecurrentScalar:
    """Tests for scalar (mean-pooled) chunked recurrent vertical attention."""

    def setup_method(self):
        self.va = ChunkedRecurrentVerticalAttention(
            hidden_dim=D, query_dim=D_Q, num_blocks=NUM_BLOCKS,
            share_topdown_keys=True, per_token=False,
        )

    def test_no_cache_output_shape(self):
        """Without cache, output shape matches input."""
        block_outputs = _make_block_outputs(2)  # [o_0, o_1, o_2]
        current = block_outputs[-1]
        out = self.va(block_outputs, current, cached_block_outputs=None, block_idx=1)
        assert out.shape == (B, T, D)

    def test_no_cache_only_bu_weights(self):
        """Without cache, only bottom-up weights are stored."""
        block_outputs = _make_block_outputs(2)
        current = block_outputs[-1]
        self.va(block_outputs, current, cached_block_outputs=None, block_idx=1)
        assert self.va.attention_weights is not None
        assert self.va.attention_weights_topdown is None

    def test_with_cache_output_shape(self):
        """With cache, output shape still matches input."""
        block_outputs = _make_block_outputs(2)  # current chunk: blocks 0,1,2
        current = block_outputs[-1]
        cache = _make_block_outputs(NUM_BLOCKS)  # full previous: blocks 0..5
        out = self.va(block_outputs, current, cached_block_outputs=cache, block_idx=1)
        assert out.shape == (B, T, D)

    def test_with_cache_has_td_weights(self):
        """With cache, top-down attention weights should be present."""
        block_outputs = _make_block_outputs(2)
        current = block_outputs[-1]
        cache = _make_block_outputs(NUM_BLOCKS)
        self.va(block_outputs, current, cached_block_outputs=cache, block_idx=1)
        assert self.va.attention_weights is not None
        assert self.va.attention_weights_topdown is not None
        # TD sources: cache[3:] = blocks 3,4,5 → 3 sources
        assert self.va.attention_weights_topdown.shape[-1] == 3

    def test_attention_weights_sum_to_one(self):
        """BU + TD attention weights should sum to 1."""
        block_outputs = _make_block_outputs(2)
        current = block_outputs[-1]
        cache = _make_block_outputs(NUM_BLOCKS)
        self.va(block_outputs, current, cached_block_outputs=cache, block_idx=1)

        bu = self.va.attention_weights
        td = self.va.attention_weights_topdown
        total = torch.cat([bu, td], dim=-1).sum(-1)
        assert torch.allclose(total, torch.ones_like(total), atol=1e-5)

    def test_no_grad_through_cache(self):
        """Gradients should not flow through cached block outputs."""
        block_outputs = [torch.randn(B, T, D, requires_grad=True) for _ in range(3)]
        current = block_outputs[-1]

        # Cache is detached
        cache = [torch.randn(B, T, D).detach() for _ in range(NUM_BLOCKS + 1)]
        for c in cache:
            assert not c.requires_grad

        out = self.va(block_outputs, current, cached_block_outputs=cache, block_idx=1)
        loss = out.sum()
        loss.backward()

        # Current chunk grads should exist
        assert current.grad is not None
        # Cache tensors should have no grad
        for c in cache:
            assert c.grad is None

    def test_different_chunk_lengths(self):
        """Cache from previous chunk can have different T than current chunk."""
        block_outputs = _make_block_outputs(2, T=10)
        current = block_outputs[-1]
        cache = _make_block_outputs(NUM_BLOCKS, T=16)  # different length
        out = self.va(block_outputs, current, cached_block_outputs=cache, block_idx=1)
        assert out.shape == (B, 10, D)

    def test_last_boundary_has_one_td(self):
        """At the last boundary (block_idx=3 for 5 blocks), B5 from the
        previous chunk is a valid top-down source (1 source)."""
        # block_idx=3 → bu has [o_0..o_4], td_start=5 → cache[5] = B5 output
        block_outputs = _make_block_outputs(4)  # [o_0, ..., o_4]
        current = block_outputs[-1]
        cache = _make_block_outputs(NUM_BLOCKS)  # [o_0, ..., o_5]
        self.va(block_outputs, current, cached_block_outputs=cache, block_idx=3)
        assert self.va.attention_weights_topdown is not None
        assert self.va.attention_weights_topdown.shape[-1] == 1

    def test_no_cache_no_td(self):
        """Without cache, there should never be top-down weights."""
        block_outputs = _make_block_outputs(4)
        current = block_outputs[-1]
        self.va(block_outputs, current, cached_block_outputs=None, block_idx=3)
        assert self.va.attention_weights_topdown is None


class TestChunkedRecurrentPerToken:
    """Tests for per-token chunked recurrent vertical attention."""

    def setup_method(self):
        self.va = ChunkedRecurrentVerticalAttention(
            hidden_dim=D, query_dim=D_Q, num_blocks=NUM_BLOCKS,
            share_topdown_keys=True, per_token=True,
        )

    def test_per_token_weight_shape(self):
        """Per-token mode should produce (B, T, N) attention weights."""
        block_outputs = _make_block_outputs(2)
        current = block_outputs[-1]
        self.va(block_outputs, current, cached_block_outputs=None, block_idx=1)
        w = self.va.attention_weights
        assert w is not None
        assert w.dim() == 3
        assert w.shape == (B, T, len(block_outputs))

    def test_per_token_with_cache(self):
        """Per-token mode with cache should include TD sources."""
        block_outputs = _make_block_outputs(2)
        current = block_outputs[-1]
        cache = _make_block_outputs(NUM_BLOCKS)
        self.va(block_outputs, current, cached_block_outputs=cache, block_idx=1)
        w_bu = self.va.attention_weights
        w_td = self.va.attention_weights_topdown
        assert w_bu.dim() == 3
        assert w_td is not None
        assert w_td.dim() == 3
        # Total sources = bu + td
        N_bu = w_bu.shape[-1]
        N_td = w_td.shape[-1]
        total = torch.cat([w_bu, w_td], dim=-1).sum(-1)
        assert torch.allclose(total, torch.ones_like(total), atol=1e-5)


class TestSeparateTopdownKeys:
    """Tests for separate W_K_topdown projection."""

    def test_separate_keys_different_params(self):
        va = ChunkedRecurrentVerticalAttention(
            hidden_dim=D, query_dim=D_Q, num_blocks=NUM_BLOCKS,
            share_topdown_keys=False, per_token=False,
        )
        # W_K and W_K_topdown should be different modules
        assert va.W_K is not va.W_K_topdown
        # But both should have same shape
        assert va.W_K.weight.shape == va.W_K_topdown.weight.shape

    def test_separate_keys_forward(self):
        va = ChunkedRecurrentVerticalAttention(
            hidden_dim=D, query_dim=D_Q, num_blocks=NUM_BLOCKS,
            share_topdown_keys=False, per_token=False,
        )
        block_outputs = _make_block_outputs(2)
        current = block_outputs[-1]
        cache = _make_block_outputs(NUM_BLOCKS)
        out = va(block_outputs, current, cached_block_outputs=cache, block_idx=1)
        assert out.shape == (B, T, D)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
