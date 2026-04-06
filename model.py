"""
HPSN Model
==========
Hierarchical Predictive Speech Network built on HuBERT-Large.

Loads pretrained HuBERT-Large (24 layers), partitions into 5 blocks,
and injects context-conditioned vertical attention at block boundaries
with subspace inhibition at the B3/B4 boundary.
"""

import torch
import torch.nn as nn
from transformers import HubertForCTC, HubertConfig
from modules import (
    ContextConditionedVerticalAttention,
    ChunkedRecurrentVerticalAttention,
    SubspaceInhibition,
)


# Default block partition for HuBERT-Large (24 layers → 5 blocks)
DEFAULT_BLOCK_BOUNDARIES = [5, 10, 15, 20, 24]  # cumulative layer counts


class HPSNHubert(nn.Module):
    """
    HuBERT-Large with Block AttnRes and Subspace Inhibition.
    
    Architecture:
        CNN Feature Extractor (frozen)
        → Feature Projection  
        → Positional Encoding
        → Block B1 [layers 0–4]  → Vertical Attention
        → Block B2 [layers 5–9]  → Vertical Attention  
        → Block B3 [layers 10–14] → ★ Subspace Inhibition ★ → Vertical Attention
        → Block B4 [layers 15–19] → Vertical Attention
        → Block B5 [layers 20–23]
        → Layer Norm → CTC Head
    
    Args:
        pretrained: HuggingFace model ID or path to pretrained HuBERT-Large
        block_boundaries: List of cumulative layer counts defining blocks
        vocab_size: CTC vocabulary size (29 = 26 letters + space + apostrophe + blank)
        query_dim: Dimension for vertical attention queries/keys
        inhibition_rank: Rank of the competition subspace
        inhibition_boundary: Which block boundary gets inhibition (0-indexed, default=2 → B3/B4)
        freeze_feature_extractor: Whether to freeze CNN feature extractor
    """
    
    def __init__(
        self,
        pretrained: str = "facebook/hubert-large-ls960-ft",
        block_boundaries: list[int] = None,
        query_dim: int = 64,
        inhibition_rank: int = 64,
        inhibition_boundary: int = 2,  # index into block boundaries (0-indexed)
        freeze_feature_extractor: bool = True,
        vertical_attention_type: str = "standard",  # "standard", "chunked", "chunked_per_token"
        chunk_size: int = 16,  # frames per chunk (~320ms at 50Hz)
        share_topdown_keys: bool = True,
    ):
        super().__init__()
        
        if block_boundaries is None:
            block_boundaries = DEFAULT_BLOCK_BOUNDARIES
        
        self.block_boundaries = block_boundaries
        self.num_blocks = len(block_boundaries)
        self.inhibition_boundary = inhibition_boundary
        self.vertical_attention_type = vertical_attention_type
        self.chunk_size = chunk_size
        
        # Load pretrained HuBERT
        hubert = HubertForCTC.from_pretrained(pretrained)  # includes trained CTC head
        self.ctc_head = hubert.lm_head                     # copy the trained projection
        config = hubert.config
        self.config = config
        self.hidden_dim = config.hidden_size  # 1024 for HuBERT-Large
        
        # Steal components from HuBERT
        self.feature_extractor = hubert.hubert.feature_extractor
        self.feature_projection = hubert.hubert.feature_projection
        self.encoder_pos_conv = hubert.hubert.encoder.pos_conv_embed
        self.encoder_layer_norm = hubert.hubert.encoder.layer_norm
        self.encoder_dropout = hubert.hubert.encoder.dropout
        self.final_dropout = hubert.dropout
        
        # Partition transformer layers into blocks
        all_layers = list(hubert.hubert.encoder.layers)
        self.blocks = nn.ModuleList()
        prev = 0
        for boundary in block_boundaries:
            self.blocks.append(nn.ModuleList(all_layers[prev:boundary]))
            prev = boundary
        
        assert prev == len(all_layers), (
            f"Block boundaries {block_boundaries} don't cover all "
            f"{len(all_layers)} layers"
        )
        
        # Novel modules: vertical attention at each block boundary (except after last)
        num_boundaries = self.num_blocks - 1  # 4 boundaries for 5 blocks
        if vertical_attention_type in ("chunked", "chunked_per_token"):
            self.vertical_attention = nn.ModuleList([
                ChunkedRecurrentVerticalAttention(
                    hidden_dim=self.hidden_dim,
                    query_dim=query_dim,
                    num_blocks=self.num_blocks,
                    share_topdown_keys=share_topdown_keys,
                    per_token=(vertical_attention_type == "chunked_per_token"),
                )
                for _ in range(num_boundaries)
            ])
        else:
            self.vertical_attention = nn.ModuleList([
                ContextConditionedVerticalAttention(
                    hidden_dim=self.hidden_dim,
                    query_dim=query_dim,
                )
                for _ in range(num_boundaries)
            ])
        
        # Subspace inhibition at one boundary
        self.inhibition = SubspaceInhibition(
            hidden_dim=self.hidden_dim,
            rank=inhibition_rank,
        )
        
        # CTC head: keep the pretrained one (already set above as hubert.lm_head)
        # self.ctc_head is already assigned on line 67
        
        # Final layer norm (from HuBERT encoder)
        # Already captured as self.encoder_layer_norm
        
        # Freeze CNN feature extractor
        if freeze_feature_extractor:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        
        # Clean up the original model
        del hubert
    
    def _run_block(
        self, 
        block: nn.ModuleList, 
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run a block of transformer layers with standard residual connections."""
        for layer in block:
            layer_outputs = layer(
                hidden_states, 
                attention_mask=attention_mask,
            )
            hidden_states = layer_outputs[0]
        return hidden_states
    
    def forward(
        self,
        input_values: torch.Tensor,          # (B, raw_audio_length)
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,  # (B, label_length) for CTC loss
        return_block_outputs: bool = False,   # for analysis / RSA
        bypass_novel: bool = False,           # skip novel modules (for pretrained baseline eval)
        rank_reg_weight: float = 0.01,
    ):
        """
        Args:
            input_values: Raw waveform, (B, T_audio), 16kHz
            attention_mask: Optional mask for padded inputs
            labels: CTC targets (character indices). If provided, computes loss.
            return_block_outputs: If True, return intermediate block representations.
        
        Returns:
            dict with keys:
                - logits: (B, T_enc, vocab_size)
                - loss: CTC loss (if labels provided)
                - block_outputs: list of (B, T_enc, D) tensors (if requested)
                - attention_weights: list of vertical attention weights (if requested)
        """
        # === Feature extraction (CNN frontend) ===
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)  # (B, T, C=512)

        # Keep the raw waveform mask for CTC input length computation.
        raw_attention_mask = attention_mask

        if attention_mask is not None:
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask,
            )

        hidden_states = self.feature_projection(extract_features)

        # Match HubertEncoder: zero padded positions, then build the mask object
        if attention_mask is not None:
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(
                1, 1, hidden_states.shape[2]
            )
            hidden_states[~expand_attention_mask] = 0.0

            from transformers.masking_utils import create_bidirectional_mask
            transformer_attention_mask = create_bidirectional_mask(
                config=self.config,
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
            )
        else:
            transformer_attention_mask = None
        
        # === Positional encoding (after masking, matching HuBERT encoder) ===
        position_embeddings = self.encoder_pos_conv(hidden_states)
        hidden_states = hidden_states + position_embeddings.to(hidden_states.device)

        # HuBERT has two encoder variants with different norm placement.
        if self.config.do_stable_layer_norm:
            hidden_states = self.encoder_dropout(hidden_states)
        else:
            hidden_states = self.encoder_layer_norm(hidden_states)
            hidden_states = self.encoder_dropout(hidden_states)
        
        # === Run blocks with vertical attention ===
        o_0 = hidden_states  # output of feature projection (block 0 input)
        block_outputs_list = [o_0]
        
        for block_idx, block in enumerate(self.blocks):
            # Run transformer layers in this block
            block_out = self._run_block(block, hidden_states, transformer_attention_mask)
            block_outputs_list.append(block_out)
            
            # Apply vertical attention at boundary (except after last block)
            if block_idx < self.num_blocks - 1:
                if bypass_novel:
                    # Straight sequential pass — pretrained baseline
                    hidden_states = block_out
                else:
                    # Apply inhibition at the designated boundary
                    if block_idx == self.inhibition_boundary:
                        block_out = self.inhibition(block_out)
                        # Update the stored output to the inhibited version
                        block_outputs_list[-1] = block_out
                    
                    # Vertical attention: compute input for next block
                    if self.vertical_attention_type in ("chunked", "chunked_per_token"):
                        hidden_states = self.vertical_attention[block_idx](
                            block_outputs_list,
                            block_out,
                            cached_block_outputs=None,  # no cache in full-utterance mode
                            block_idx=block_idx,
                        )
                    else:
                        hidden_states = self.vertical_attention[block_idx](
                            block_outputs_list,  # all outputs so far
                            block_out,           # current block output
                        )
            else:
                hidden_states = block_out

        if self.config.do_stable_layer_norm:
            hidden_states = self.encoder_layer_norm(hidden_states)
        
        # === CTC head ===
        hidden_states = self.final_dropout(hidden_states)
        logits = self.ctc_head(hidden_states)  # (B, T, V)
        
        # === Compute loss if labels provided ===
        result = {"logits": logits}
        
        if labels is not None:
            log_probs = nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)
            # (T, B, V) for CTC loss
            
            input_lengths = torch.full(
                (logits.shape[0],), logits.shape[1], 
                dtype=torch.long, device=logits.device,
            )
            if raw_attention_mask is not None:
                input_lengths = self._get_feat_extract_output_lengths(
                    raw_attention_mask.sum(-1).long()
                )
            
            # Label lengths (non-pad)
            label_mask = labels >= 0
            label_lengths = label_mask.sum(-1)
            labels_for_ctc = labels.clamp(min=0)
            
            ctc_loss = nn.functional.ctc_loss(
                log_probs, labels_for_ctc, input_lengths, label_lengths,
                blank=0, reduction="mean", zero_infinity=True,
            )
            
            # Add rank regularization
            rank_loss = self.inhibition.rank_regularization()
            
            result["loss"] = ctc_loss + rank_reg_weight * rank_loss
            result["ctc_loss"] = ctc_loss.detach()
            result["rank_loss"] = rank_loss.detach()
        
        if return_block_outputs:
            result["block_outputs"] = block_outputs_list
            result["attention_weights"] = [
                va.attention_weights for va in self.vertical_attention
            ]
            result["inhibition_effective_rank"] = self.inhibition.effective_rank
        
        return result
    
    def forward_chunked(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_block_outputs: bool = False,
        cache: list[torch.Tensor] | None = None,
        rank_reg_weight: float = 0.01,
    ):
        """
        Chunked forward pass with recurrent top-down vertical attention.

        Splits the encoded sequence into chunks of self.chunk_size frames.
        Each chunk runs all blocks with vertical attention that can attend to
        both lower blocks (current chunk) and higher blocks (previous chunk,
        via detached cache). CTC loss is computed on the full concatenated logits.

        Args:
            input_values: Raw waveform, (B, T_audio)
            attention_mask: Optional mask for padded inputs
            labels: CTC targets. If provided, computes loss.
            return_block_outputs: If True, return per-chunk block outputs and attention info.
            cache: Previous chunk's block outputs (detached). None for first call.

        Returns:
            dict with same keys as forward(), plus:
                - cache: list of block output tensors (detached) for next call
                - chunk_attention_weights: list of per-chunk attention weight dicts
        """
        # === Feature extraction (same as forward) ===
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        raw_attention_mask = attention_mask

        if attention_mask is not None:
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask,
            )

        hidden_states = self.feature_projection(extract_features)

        if attention_mask is not None:
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(
                1, 1, hidden_states.shape[2]
            )
            hidden_states[~expand_attention_mask] = 0.0

            from transformers.masking_utils import create_bidirectional_mask
            transformer_attention_mask = create_bidirectional_mask(
                config=self.config,
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
            )
        else:
            transformer_attention_mask = None

        position_embeddings = self.encoder_pos_conv(hidden_states)
        hidden_states = hidden_states + position_embeddings.to(hidden_states.device)

        if self.config.do_stable_layer_norm:
            hidden_states = self.encoder_dropout(hidden_states)
        else:
            hidden_states = self.encoder_layer_norm(hidden_states)
            hidden_states = self.encoder_dropout(hidden_states)

        # === Chunk the encoded sequence ===
        B, T_full, D = hidden_states.shape
        chunk_size = self.chunk_size

        # Split into chunks along time dimension
        num_chunks = (T_full + chunk_size - 1) // chunk_size
        chunks = []
        mask_chunks = []
        for i in range(num_chunks):
            t_start = i * chunk_size
            t_end = min((i + 1) * chunk_size, T_full)
            chunks.append(hidden_states[:, t_start:t_end, :])
            if transformer_attention_mask is not None:
                # attention_mask is (B, 1, T, T) or (B, 1, 1, T) — slice spatial dims
                if transformer_attention_mask.dim() == 4:
                    mask_chunks.append(
                        transformer_attention_mask[:, :, t_start:t_end, t_start:t_end]
                    )
                else:
                    mask_chunks.append(None)
            else:
                mask_chunks.append(None)

        # === Process chunks sequentially with cache ===
        all_logits_chunks = []
        all_block_outputs = []  # list of lists, one per chunk
        all_chunk_attn = []
        current_cache = cache

        for chunk_idx in range(num_chunks):
            chunk_hidden = chunks[chunk_idx]
            chunk_mask = mask_chunks[chunk_idx]

            o_0 = chunk_hidden
            block_outputs_list = [o_0]

            hidden = chunk_hidden
            for block_idx, block in enumerate(self.blocks):
                block_out = self._run_block(block, hidden, chunk_mask)
                block_outputs_list.append(block_out)

                if block_idx < self.num_blocks - 1:
                    # Apply inhibition at the designated boundary
                    if block_idx == self.inhibition_boundary:
                        block_out = self.inhibition(block_out)
                        block_outputs_list[-1] = block_out

                    # Vertical attention with top-down cache
                    hidden = self.vertical_attention[block_idx](
                        block_outputs_list,
                        block_out,
                        cached_block_outputs=current_cache,
                        block_idx=block_idx,
                    )
                else:
                    hidden = block_out

            # Final layer norm for this chunk
            if self.config.do_stable_layer_norm:
                chunk_final = self.encoder_layer_norm(hidden)
            else:
                chunk_final = hidden

            chunk_final = self.final_dropout(chunk_final)
            chunk_logits = self.ctc_head(chunk_final)
            all_logits_chunks.append(chunk_logits)

            # Update cache: detach all block outputs for next chunk
            current_cache = [o.detach() for o in block_outputs_list]

            if return_block_outputs:
                all_block_outputs.append(block_outputs_list)
                all_chunk_attn.append({
                    f"boundary_{i}": {
                        "bu": self.vertical_attention[i].attention_weights,
                        "td": (self.vertical_attention[i].attention_weights_topdown
                               if hasattr(self.vertical_attention[i], "attention_weights_topdown")
                               else None),
                    }
                    for i in range(len(self.vertical_attention))
                })

        # === Concatenate logits across chunks ===
        logits = torch.cat(all_logits_chunks, dim=1)  # (B, T_full, V)

        result = {"logits": logits, "cache": current_cache}

        # === CTC loss (same as forward) ===
        if labels is not None:
            log_probs = nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)

            input_lengths = torch.full(
                (logits.shape[0],), logits.shape[1],
                dtype=torch.long, device=logits.device,
            )
            if raw_attention_mask is not None:
                input_lengths = self._get_feat_extract_output_lengths(
                    raw_attention_mask.sum(-1).long()
                )

            label_mask = labels >= 0
            label_lengths = label_mask.sum(-1)
            labels_for_ctc = labels.clamp(min=0)

            ctc_loss = nn.functional.ctc_loss(
                log_probs, labels_for_ctc, input_lengths, label_lengths,
                blank=0, reduction="mean", zero_infinity=True,
            )

            rank_loss = self.inhibition.rank_regularization()

            result["loss"] = ctc_loss + rank_reg_weight * rank_loss
            result["ctc_loss"] = ctc_loss.detach()
            result["rank_loss"] = rank_loss.detach()

        if return_block_outputs:
            result["chunk_block_outputs"] = all_block_outputs
            result["chunk_attention_weights"] = all_chunk_attn
            result["inhibition_effective_rank"] = self.inhibition.effective_rank

        return result

    def _get_feat_extract_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Compute output lengths after CNN feature extractor."""
        # HuBERT CNN: kernel_sizes=[10,3,3,3,3,2,2], strides=[5,2,2,2,2,2,2]
        for kernel_size, stride in zip(
            [10, 3, 3, 3, 3, 2, 2], 
            [5, 2, 2, 2, 2, 2, 2],
        ):
            input_lengths = (input_lengths - kernel_size) // stride + 1
        return input_lengths

    def _get_feature_vector_attention_mask(
        self,
        feature_vector_length: int,
        attention_mask: torch.LongTensor,
    ) -> torch.BoolTensor:
        """Match HubertModel._get_feature_vector_attention_mask exactly."""
        output_lengths = self._get_feat_extract_output_lengths(
            attention_mask.sum(-1)
        ).to(torch.long)
        batch_size = attention_mask.shape[0]

        feature_attention_mask = torch.zeros(
            (batch_size, feature_vector_length),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        feature_attention_mask[
            (torch.arange(batch_size, device=attention_mask.device), output_lengths - 1)
        ] = 1
        feature_attention_mask = feature_attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return feature_attention_mask
    
    def get_param_groups(self, lr_backbone: float = 3e-5, lr_novel: float = 1e-4):
        """
        Separate parameter groups with different learning rates.
        Novel modules get higher LR than the pretrained backbone.
        """
        backbone_params = []
        novel_params = []
        
        novel_module_names = {"vertical_attention", "inhibition", "ctc_head"}
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(nm in name for nm in novel_module_names):
                novel_params.append(param)
            else:
                backbone_params.append(param)
        
        return [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": novel_params, "lr": lr_novel},
        ]
    
    def freeze_backbone(self):
        """Freeze all pretrained parameters (for Stage 1 training)."""
        for name, param in self.named_parameters():
            if "vertical_attention" in name or "inhibition" in name or "ctc_head" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone (for Stage 2 training). CNN stays frozen."""
        for name, param in self.named_parameters():
            if "feature_extractor" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
