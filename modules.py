"""
HPSN Novel Modules
==================
Context-Conditioned Vertical Attention (Block AttnRes) and
Subspace Inhibition — the two core contributions.

Reference: HPSN proposal Sections 3.4 and 3.5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextConditionedVerticalAttention(nn.Module):
    """
    Context-conditioned vertical attention at a block boundary.
    
    Unlike vanilla AttnRes (fixed pseudo-query), the query is derived
    from the preceding block's output, making it input-dependent.
    
    Given block outputs [o_0, o_1, ..., o_{b-1}]:
      1. query = W_Q @ MeanPool(o_{b-1})           (context-conditioned)
      2. keys  = W_K @ MeanPool(o_j) for j < b      
      3. alpha = softmax(q^T k / sqrt(d_q))         (scalar per source block)
      4. context = sum_j alpha_j * o_j               (Txd weighted sum)
      5. output = context + o_{b-1}                  (residual)
    
    Args:
        hidden_dim: Model hidden dimension (1024 for HuBERT-Large)
        query_dim: Dimension of the vertical attention query/key space (default: 64)
    """
    
    def __init__(self, hidden_dim: int = 1024, query_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query_dim = query_dim
        
        self.W_Q = nn.Linear(hidden_dim, query_dim, bias=False)
        self.W_K = nn.Linear(hidden_dim, query_dim, bias=False)
        
        # For logging / analysis
        self._last_attention_weights = None
    
    def forward(
        self, 
        block_outputs: list[torch.Tensor],  # [o_0, o_1, ..., o_{b-1}], each (B, T, D)
        current_block_output: torch.Tensor,  # o_{b-1}, shape (B, T, D)
    ) -> torch.Tensor:
        """
        Returns: input for the next block, shape (B, T, D)
        """
        # Query from current block output (context-conditioned)
        # MeanPool over time → (B, D) → project → (B, d_q)
        query = self.W_Q(current_block_output.mean(dim=1))  # (B, d_q)
        
        # Keys from all preceding block outputs
        keys = []
        for o_j in block_outputs:
            k_j = self.W_K(o_j.mean(dim=1))  # (B, d_q)
            keys.append(k_j)
        keys = torch.stack(keys, dim=1)  # (B, num_sources, d_q)
        
        # Attention weights (scalar per source block)
        # query: (B, d_q) → (B, 1, d_q)
        scores = torch.bmm(
            query.unsqueeze(1), keys.transpose(1, 2)
        ).squeeze(1) / (self.query_dim ** 0.5)  # (B, num_sources)
        
        alpha = F.softmax(scores, dim=-1)  # (B, num_sources)
        self._last_attention_weights = alpha.detach()
        
        # Weighted sum of block outputs
        # Stack sources: (B, num_sources, T, D)
        stacked = torch.stack(block_outputs, dim=1)
        # alpha: (B, num_sources, 1, 1) for broadcasting
        context = (alpha.unsqueeze(-1).unsqueeze(-1) * stacked).sum(dim=1)  # (B, T, D)
        
        # Residual connection with preceding block
        return context + current_block_output
    
    @property
    def attention_weights(self):
        """Last computed attention weights for analysis."""
        return self._last_attention_weights


class ChunkedRecurrentVerticalAttention(nn.Module):
    """
    Chunked recurrent vertical attention with true top-down feedback.

    At each chunk, block b attends to:
      - Bottom-up: blocks 0..b-1 from the CURRENT chunk (same as standard)
      - Top-down:  blocks b..N from the PREVIOUS chunk (cached, detached)

    This gives each block access to higher-level interpretations from the
    recent past, mirroring incremental speech processing where higher
    cortical representations influence ongoing lower-level processing.

    The cache is detached (no gradient through time), analogous to
    truncated BPTT. First chunk has no cache → pure bottom-up.

    Args:
        hidden_dim: Model hidden dimension (1024 for HuBERT-Large)
        query_dim: Dimension of the vertical attention query/key space (default: 64)
        num_blocks: Total number of blocks in the model (for top-down source count)
        share_topdown_keys: If True, use same W_K for bottom-up and top-down keys.
            If False, use a separate W_K_topdown projection.
        per_token: If True, compute per-token attention weights (no mean-pooling).
            Uses t-1 shifted queries within each chunk for temporal dynamics.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        query_dim: int = 64,
        num_blocks: int = 5,
        share_topdown_keys: bool = True,
        per_token: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query_dim = query_dim
        self.num_blocks = num_blocks
        self.per_token = per_token

        self.W_Q = nn.Linear(hidden_dim, query_dim, bias=False)
        self.W_K = nn.Linear(hidden_dim, query_dim, bias=False)

        if not share_topdown_keys:
            self.W_K_topdown = nn.Linear(hidden_dim, query_dim, bias=False)
        else:
            self.W_K_topdown = self.W_K

        if per_token:
            # Learned initial query for t=0 within each chunk
            self.q_init = nn.Parameter(torch.randn(query_dim) * 0.02)

        # For logging / analysis
        self._last_attention_weights_bu = None
        self._last_attention_weights_td = None

    def forward(
        self,
        block_outputs: list[torch.Tensor],          # current chunk: [o_0, ..., o_b], each (B, T, D)
        current_block_output: torch.Tensor,          # o_b from current chunk, (B, T, D)
        cached_block_outputs: list[torch.Tensor] | None = None,  # previous chunk: [o_0, ..., o_N], each (B, L, D)
        block_idx: int = 0,                          # which boundary (0-indexed)
    ) -> torch.Tensor:
        """
        Args:
            block_outputs: All block outputs computed so far in current chunk
                           (blocks 0..b), each (B, T, D).
            current_block_output: Output of block b in current chunk, (B, T, D).
            cached_block_outputs: All block outputs from previous chunk (detached),
                                  blocks 0..N, each (B, L, D). None for first chunk.
            block_idx: Index of the current boundary (0-indexed).

        Returns:
            Input for the next block, shape (B, T, D).
        """
        B, T, D = current_block_output.shape
        bu_sources = block_outputs  # blocks 0..b from current chunk

        # Top-down sources: blocks (b+1)..N from previous chunk
        td_sources = []
        if cached_block_outputs is not None:
            # block_idx is 0-indexed boundary; block_outputs has (block_idx+2) entries
            # (o_0, o_1, ..., o_{block_idx+1}). Top-down = cached blocks from
            # (block_idx+1) onward (i.e. the current block's level and above).
            td_start = block_idx + 2  # skip blocks already in bu_sources
            if td_start < len(cached_block_outputs):
                td_sources = cached_block_outputs[td_start:]

        if self.per_token:
            return self._forward_per_token(
                current_block_output, bu_sources, td_sources, B, T,
            )
        else:
            return self._forward_scalar(
                current_block_output, bu_sources, td_sources, B, T,
            )

    def _forward_scalar(self, current_block_output, bu_sources, td_sources, B, T):
        """Scalar-per-source attention weights (mean-pooled, like standard)."""
        # Query from current block (mean-pooled)
        query = self.W_Q(current_block_output.mean(dim=1))  # (B, d_q)

        # Bottom-up keys
        bu_keys = torch.stack(
            [self.W_K(o.mean(dim=1)) for o in bu_sources], dim=1
        )  # (B, N_bu, d_q)

        # Top-down keys
        if td_sources:
            td_keys = torch.stack(
                [self.W_K_topdown(o.mean(dim=1)) for o in td_sources], dim=1
            )  # (B, N_td, d_q)
            all_keys = torch.cat([bu_keys, td_keys], dim=1)  # (B, N_bu+N_td, d_q)
        else:
            all_keys = bu_keys

        # Attention weights
        scores = torch.bmm(
            query.unsqueeze(1), all_keys.transpose(1, 2)
        ).squeeze(1) / (self.query_dim ** 0.5)  # (B, N_total)
        alpha = F.softmax(scores, dim=-1)  # (B, N_total)

        N_bu = len(bu_sources)

        # Store separated attention weights for diagnostics
        self._last_attention_weights_bu = alpha[:, :N_bu].detach()
        self._last_attention_weights_td = alpha[:, N_bu:].detach() if td_sources else None

        # Weighted sum of all sources
        # Bottom-up: (B, N_bu, T, D)
        all_outputs = list(bu_sources)
        if td_sources:
            # Top-down sources may have different T (previous chunk length L).
            # Interpolate to match current chunk T if needed.
            for o_td in td_sources:
                if o_td.shape[1] != T:
                    # Linear interpolation along time axis
                    o_td = F.interpolate(
                        o_td.transpose(1, 2), size=T, mode="linear", align_corners=False
                    ).transpose(1, 2)
                all_outputs.append(o_td)

        stacked = torch.stack(all_outputs, dim=1)  # (B, N_total, T, D)
        context = (alpha.unsqueeze(-1).unsqueeze(-1) * stacked).sum(dim=1)  # (B, T, D)

        return context + current_block_output

    def _forward_per_token(self, current_block_output, bu_sources, td_sources, B, T):
        """Per-token attention weights with t-1 shifted queries."""
        # Shift current block output by 1 position for causal query
        # q[t] = W_Q(o_{b}[t-1]), q[0] = q_init
        shifted = torch.cat([
            self.q_init.unsqueeze(0).unsqueeze(0).expand(B, 1, -1),  # (B, 1, d_q)
            self.W_Q(current_block_output[:, :-1, :]),                # (B, T-1, d_q)
        ], dim=1)  # (B, T, d_q)

        # Bottom-up keys per token: (B, T, N_bu, d_q)
        bu_keys = torch.stack(
            [self.W_K(o) for o in bu_sources], dim=2
        )  # (B, T, N_bu, d_q)

        if td_sources:
            td_keys_list = []
            td_vals_list = []
            for o_td in td_sources:
                if o_td.shape[1] != T:
                    o_td = F.interpolate(
                        o_td.transpose(1, 2), size=T, mode="linear", align_corners=False
                    ).transpose(1, 2)
                td_keys_list.append(self.W_K_topdown(o_td))
                td_vals_list.append(o_td)
            td_keys = torch.stack(td_keys_list, dim=2)  # (B, T, N_td, d_q)
            all_keys = torch.cat([bu_keys, td_keys], dim=2)  # (B, T, N_total, d_q)
        else:
            all_keys = bu_keys
            td_vals_list = []

        # Per-token attention: (B, T, N_total)
        scores = torch.einsum("btd,btnd->btn", shifted, all_keys) / (self.query_dim ** 0.5)
        alpha = F.softmax(scores, dim=-1)  # (B, T, N_total)

        N_bu = len(bu_sources)
        self._last_attention_weights_bu = alpha[:, :, :N_bu].detach()
        self._last_attention_weights_td = alpha[:, :, N_bu:].detach() if td_sources else None

        # Weighted sum: build (B, T, N_total, D) values tensor
        all_vals = list(bu_sources) + td_vals_list
        stacked = torch.stack(all_vals, dim=2)  # (B, T, N_total, D)
        context = torch.einsum("btn,btnd->btd", alpha, stacked)  # (B, T, D)

        return context + current_block_output

    @property
    def attention_weights(self):
        """Last bottom-up attention weights (for compatibility with standard module)."""
        return self._last_attention_weights_bu

    @property
    def attention_weights_topdown(self):
        """Last top-down attention weights for analysis."""
        return self._last_attention_weights_td


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
        lambda_init: float = 0.01,
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
    
    def rank_regularization(self, r_min: float = 5.0, r_max: float = 50.0) -> torch.Tensor:
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
