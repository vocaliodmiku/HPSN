"""
HPSN Novel Modules
==================
RefineLayer (cross-attention refinement) and Subspace Inhibition.

RefineLayer: Each block's output is refined by attending to all block
outputs (both already-refined lower blocks and raw higher blocks).

SubspaceInhibition: Competitive suppression via low-rank projection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RefineLayer(nn.Module):
    """
    Cross-attention refinement module for one hierarchical level.

    Takes a query (one block's output) and a context set (all block outputs,
    mix of refined and raw) and produces a refined representation via
    multi-head cross-attention with a gated residual.

    The gate is initialized near zero so the output starts nearly identical
    to the raw block output, preserving pretrained representations.

    Args:
        hidden_dim: Model hidden dimension (1024 for HuBERT-Large)
        num_heads: Number of attention heads (default: 4)
    """

    def __init__(self, hidden_dim: int = 1024, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True,
        )
        # Gate initialized so refinement participates from the start.
        # sigmoid(-1) ≈ 0.27 — large enough for meaningful gradient flow,
        # small enough that the pretrained signal still dominates early on.
        self.log_gate = nn.Parameter(torch.tensor(-1.0))

        # For logging / analysis
        self._last_attention_weights = None

    @property
    def gate(self):
        return torch.sigmoid(self.log_gate)

    def forward(
        self,
        query: torch.Tensor,               # (B, T, D) — one block's output
        context_list: list[torch.Tensor],   # list of N tensors, each (B, T, D)
    ) -> torch.Tensor:
        """
        Args:
            query: The block output to refine, shape (B, T, D).
            context_list: All block representations (refined + raw),
                          list of N tensors each (B, T, D).
        Returns:
            Refined output, shape (B, T, D).
        """
        B, T, D = query.shape
        N = len(context_list)

        # Stack context: (B, N*T, D)
        context = torch.cat(context_list, dim=1)  # (B, N*T, D)

        # Normalize
        q = self.norm_q(query)          # (B, T, D)
        kv = self.norm_kv(context)      # (B, N*T, D)

        # Cross-attention
        attn_out, attn_weights = self.cross_attn(
            q, kv, kv, need_weights=True, average_attn_weights=False,
        )  # attn_out: (B, T, D), attn_weights: (B, num_heads, T, N*T)

        # Store per-block attention mass for analysis
        # Reshape weights to (B, num_heads, T, N, T_per_block) then sum over T_per_block
        if attn_weights is not None:
            with torch.no_grad():
                # attn_weights: (B, num_heads, T_q, N*T_k)
                w = attn_weights.reshape(B, self.num_heads, T, N, T)
                # Sum over within-block time positions → (B, num_heads, T_q, N)
                block_mass = w.sum(dim=-1)
                # Average over heads and query positions → (B, N)
                self._last_attention_weights = block_mass.mean(dim=1).mean(dim=1)

        # Gated residual
        return query + self.gate * attn_out


class SubspaceInhibition(nn.Module):
    """
    Subspace inhibition module for competitive suppression.
    
    Applied at a single block boundary (default: B3/B4).
    Projects representations into a low-rank "competition subspace",
    identifies dominant competitors via softmax, and subtracts
    the competitor signal.
    
    Steps:
      1. z = P^T @ o_t         (project to competition subspace, R^r)
      2. s = softmax(z / tau)   (identify competitors)
      3. inhib = P @ s          (project back to full space, R^d)
      4. o_tilde = o - lambda * inhib  (subtract competitor signal)
    
    Args:
        hidden_dim: Model hidden dimension (1024)
        rank: Dimension of competition subspace (default: 64)
        tau_init: Initial temperature (default: 1.0)
        lambda_init: Initial inhibition strength (default: 0.01)
    """
    
    def __init__(
        self, 
        hidden_dim: int = 1024, 
        rank: int = 64,
        tau_init: float = 1.0,
        lambda_init: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank
        
        # Competition subspace projection (d x r)
        self.P = nn.Parameter(torch.randn(hidden_dim, rank) * 0.01)
        
        # Learnable scalars
        self.log_tau = nn.Parameter(torch.tensor(tau_init).log())
        self.log_lambda = nn.Parameter(torch.tensor(lambda_init).log())
        
        # For monitoring
        self._last_inhibition_strength = None
    
    @property
    def tau(self):
        return self.log_tau.exp()
    
    @property
    def lambda_(self):
        return self.log_lambda.exp()
    
    def forward(self, block_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            block_output: (B, T, D) — output of the target block
        Returns:
            inhibited output: (B, T, D)
        """
        # Step 1: Project to competition subspace
        z = block_output @ self.P  # (B, T, r)
        
        # Step 2: Identify competitor activation pattern
        s = F.softmax(z / self.tau, dim=-1)  # (B, T, r)
        
        # Step 3: Project back to full space
        inhib = s @ self.P.T  # (B, T, D)
        
        # Step 4: Subtractive inhibition
        inhibited = block_output - self.lambda_ * inhib
        
        self._last_inhibition_strength = self.lambda_.detach().item()
        
        return inhibited
    
    def rank_regularization(self, r_min: float = 15.0, r_max: float = 50.0) -> torch.Tensor:
        """
        Effective rank regularization for P using participation ratio.
        Penalizes if effective rank is outside [r_min, r_max].

        erank(P) = tr(G)^2 / tr(G^2),  G = P^T P
        Decomposition-free, fully differentiable, numerically stable.
        """
        P = self.P.float()
        gram = P.t() @ P
        tr_G = gram.trace()
        tr_G2 = (gram * gram).sum()  # tr(G^2) = ||G||_F^2
        effective_rank = tr_G ** 2 / (tr_G2 + 1e-12)

        # Hinge loss: penalize only when outside bounds
        loss = (
            F.relu(r_min - effective_rank) +
            F.relu(effective_rank - r_max)
        )
        return loss

    @property
    def effective_rank(self) -> float:
        """Current effective rank of P (for monitoring)."""
        with torch.no_grad():
            gram = self.P.float().t() @ self.P.float()
            tr_G = gram.trace()
            tr_G2 = (gram * gram).sum()
            return (tr_G ** 2 / (tr_G2 + 1e-12)).item()
