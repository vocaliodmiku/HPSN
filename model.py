"""
HPSN Model (v2)
===============
Hierarchical Predictive Speech Network built on HuBERT-Large.

Two-pass architecture:
  Pass 1 (no_grad): Run pretrained HuBERT blocks to extract hierarchical
                     representations [o0, o1, o2, o3, o4].
  Pass 2 (grad):    Sequentially refine each level via cross-attention
                     RefineLayer modules with full-context access.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import HubertForCTC, HubertConfig
from modules import RefineLayer, SubspaceInhibition


# Default block partition for HuBERT-Large (24 layers → 5 blocks)
DEFAULT_BLOCK_BOUNDARIES = [5, 10, 15, 20, 24]  # cumulative layer counts


class HPSNHubert(nn.Module):
    """
    HuBERT-Large with Two-Pass Hierarchical Refinement.
    
    Architecture:
        Pass 1 (frozen, no_grad):
            CNN Feature Extractor → Feature Projection → Positional Encoding
            → Block B0 [layers 0–4] → Block B1 [layers 5–9] → Block B2 [layers 10–14]
            → Block B3 [layers 15–19] → Block B4 [layers 20–23]
            Produces: [o0, o1, o2, o3, o4]

        Pass 2 (trainable):
            r0 = RefineLayer_0(query=o0, context=[o0, o1, o2, o3, o4])
            r1 = RefineLayer_1(query=o1, context=[r0, o1, o2, o3, o4])
            r2 = RefineLayer_2(query=Inhibit(o2), context=[r0, r1, o2, o3, o4])
            r3 = RefineLayer_3(query=o3, context=[r0, r1, r2, o3, o4])
            r4 = RefineLayer_4(query=o4, context=[r0, r1, r2, r3, o4])
            → Layer Norm → CTC Head on r4
    
    Args:
        pretrained: HuggingFace model ID or path to pretrained HuBERT-Large
        block_boundaries: List of cumulative layer counts defining blocks
        refine_num_heads: Number of attention heads in RefineLayer (default: 4)
        inhibition_rank: Rank of the competition subspace
        inhibition_boundary: Which level gets inhibition (0-indexed, default=2)
        freeze_feature_extractor: Whether to freeze CNN feature extractor
    """
    
    def __init__(
        self,
        pretrained: str = "facebook/hubert-large-ls960-ft",
        block_boundaries: list[int] = None,
        refine_num_heads: int = 4,
        inhibition_rank: int = 64,
        inhibition_boundary: int = 2,
        freeze_feature_extractor: bool = True,
    ):
        super().__init__()
        
        if block_boundaries is None:
            block_boundaries = DEFAULT_BLOCK_BOUNDARIES
        
        self.block_boundaries = block_boundaries
        self.num_blocks = len(block_boundaries)
        self.inhibition_boundary = inhibition_boundary
        
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
        
        # Novel modules: one RefineLayer per block level
        num_levels = self.num_blocks  # 5 levels (o0 through o4)
        self.refine_layers = nn.ModuleList([
            RefineLayer(
                hidden_dim=self.hidden_dim,
                num_heads=refine_num_heads,
            )
            for _ in range(num_levels)
        ])
        
        # Subspace inhibition at one boundary
        self.inhibition = SubspaceInhibition(
            hidden_dim=self.hidden_dim,
            rank=inhibition_rank,
        )

        # Inter-level CPC projection heads: refined[b+1] predicts raw[b]
        # 4 pairs: (r1→o0), (r2→o1), (r3→o2), (r4→o3)
        self.cpc_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            for _ in range(num_levels - 1)
        ])
        
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
        bypass_novel: bool = False,           # skip Pass 2 (pretrained baseline eval)
        rank_reg_weight: float = 0.01,
        denoise_std: float = 0.0,            # noise std for lower-level denoising
        denoise_weight: float = 0.0,         # weight for denoising loss
        sharpen_weight: float = 0.0,         # weight for contrastive sharpening loss
        sharpen_temp: float = 0.1,           # temperature for sharpening loss
        cpc_weight: float = 0.0,            # weight for inter-level CPC loss
        cpc_temp: float = 0.1,              # temperature for CPC InfoNCE
    ):
        """
        Two-pass forward.

        Pass 1 (no_grad): Run all HuBERT blocks → [o0, o1, ..., o4]
        Pass 2 (grad):    Refine each level sequentially → [r0, r1, ..., r4]
        CTC on r4 (or o4 if bypass_novel).

        Returns:
            dict with keys:
                - logits: (B, T_enc, vocab_size)
                - loss: CTC loss + rank reg (if labels provided)
                - raw_block_outputs: list of (B, T, D) (if return_block_outputs)
                - refined_block_outputs: list of (B, T, D) (if return_block_outputs)
                - refine_attention_weights: list of (B, N) (if return_block_outputs)
        """
        # === Feature extraction (CNN frontend) ===
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)  # (B, T, C=512)

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

        # === Positional encoding ===
        position_embeddings = self.encoder_pos_conv(hidden_states)
        hidden_states = hidden_states + position_embeddings.to(hidden_states.device)

        if self.config.do_stable_layer_norm:
            hidden_states = self.encoder_dropout(hidden_states)
        else:
            hidden_states = self.encoder_layer_norm(hidden_states)
            hidden_states = self.encoder_dropout(hidden_states)

        # ============================================================
        # Pass 1: Run all blocks under no_grad to get raw block outputs
        # ============================================================
        with torch.no_grad():
            h = hidden_states
            raw_block_outputs = [h]  # o0 = projected features
            for block in self.blocks:
                h = self._run_block(block, h, transformer_attention_mask)
                raw_block_outputs.append(h)
        # raw_block_outputs = [o0, o1, o2, o3, o4], all detached (no grad)
        # Note: o0 is the projected features (input to Block 0)
        #       o1..o4 are the outputs of Blocks 0..3
        #       o5 is the output of Block 4 (not present — we have num_blocks entries after o0)
        # Actually: len(raw_block_outputs) = num_blocks + 1 = 6
        # [o_feat, o_block0, o_block1, o_block2, o_block3, o_block4]
        # For the refinement, we use indices 1..5 (the 5 block outputs)
        # to align with the paper notation where o_b = output of block b.

        clean_targets = {}
        pre_inhib = None
        post_inhib = None

        if bypass_novel:
            # Use raw output of last block directly
            final_hidden = raw_block_outputs[-1]
            refined = None
        else:
            # ============================================================
            # Pass 2: Sequential refinement with full-context access
            # ============================================================
            num_levels = self.num_blocks  # 5
            o = [raw_block_outputs[i + 1] for i in range(num_levels)]

            # Inject noise into lower (acoustic) levels during training
            # to force top-down refinement to actively denoise
            if denoise_std > 0 and self.training:
                for lvl in (0, 1):
                    clean_targets[lvl] = o[lvl]  # save reference before corruption
                    o[lvl] = o[lvl] + torch.randn_like(o[lvl]) * denoise_std

            refined = []
            for b in range(num_levels):
                context = list(refined) + o[b:]

                query = o[b]
                if b == self.inhibition_boundary:
                    pre_inhib = query
                    query = self.inhibition(query)
                    post_inhib = query

                r_b = self.refine_layers[b](query=query, context_list=context)
                refined.append(r_b)

            final_hidden = refined[-1]  # r4

        # === Final layer norm + CTC head ===
        if self.config.do_stable_layer_norm:
            final_hidden = self.encoder_layer_norm(final_hidden)

        final_hidden = self.final_dropout(final_hidden)
        logits = self.ctc_head(final_hidden)  # (B, T, V)

        # === Compute loss ===
        result = {"logits": logits}

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
            total_loss = ctc_loss + rank_reg_weight * rank_loss

            # Denoising loss: refined levels should recover clean representations
            denoise_loss = torch.tensor(0.0, device=logits.device)
            if clean_targets and denoise_weight > 0 and refined is not None:
                for lvl, clean in clean_targets.items():
                    denoise_loss = denoise_loss + F.mse_loss(refined[lvl], clean)
                denoise_loss = denoise_loss / len(clean_targets)
                total_loss = total_loss + denoise_weight * denoise_loss

            # Sharpening loss: post-inhibition features should cluster by phoneme
            sharpen_loss = torch.tensor(0.0, device=logits.device)
            if sharpen_weight > 0 and post_inhib is not None:
                with torch.no_grad():
                    pt_hidden = raw_block_outputs[-1]
                    if self.config.do_stable_layer_norm:
                        pt_hidden = self.encoder_layer_norm(pt_hidden)
                    pseudo_labels = self.ctc_head(pt_hidden).argmax(dim=-1)
                sharpen_loss = self._compute_sharpen_loss(
                    post_inhib, pseudo_labels, sharpen_temp,
                )
                total_loss = total_loss + sharpen_weight * sharpen_loss

            # Inter-level CPC loss: refined[b+1] predicts raw[b] via InfoNCE
            cpc_loss = torch.tensor(0.0, device=logits.device)
            if cpc_weight > 0 and refined is not None:
                cpc_loss = self._compute_cpc_loss(refined, o, cpc_temp)
                total_loss = total_loss + cpc_weight * cpc_loss

            result["loss"] = total_loss
            result["ctc_loss"] = ctc_loss.detach()
            result["rank_loss"] = rank_loss.detach()
            result["denoise_loss"] = denoise_loss.detach()
            result["sharpen_loss"] = sharpen_loss.detach()
            result["cpc_loss"] = cpc_loss.detach()

        if return_block_outputs:
            result["raw_block_outputs"] = raw_block_outputs
            result["refined_block_outputs"] = refined if not bypass_novel else None
            result["refine_attention_weights"] = [
                rl._last_attention_weights for rl in self.refine_layers
            ] if not bypass_novel else None
            result["inhibition_effective_rank"] = self.inhibition.effective_rank

        return result

    def _compute_sharpen_loss(
        self, post_inhib, pseudo_labels, temperature=0.1, max_samples=512,
    ):
        """
        Supervised contrastive loss: post-inhibition features should cluster
        by phoneme identity (from CTC pseudo-labels).

        Uses SupCon (Khosla et al. 2020): for each anchor frame, maximize
        log-probability of same-label frames relative to all other frames.
        """
        B, T, D = post_inhib.shape
        feats = post_inhib.reshape(B * T, D)
        labs = pseudo_labels.reshape(B * T)

        # Filter out CTC blank frames (id=0)
        mask = labs != 0
        feats = feats[mask]
        labs = labs[mask]

        if feats.shape[0] < 10 or labs.unique().numel() < 2:
            return torch.tensor(0.0, device=post_inhib.device)

        # Subsample for memory efficiency
        N = min(feats.shape[0], max_samples)
        if feats.shape[0] > N:
            idx = torch.randperm(feats.shape[0], device=feats.device)[:N]
            feats = feats[idx]
            labs = labs[idx]

        feats = F.normalize(feats, dim=-1)
        sim = feats @ feats.T / temperature  # (N, N)

        # Positive mask: same label, exclude self
        pos_mask = (labs.unsqueeze(0) == labs.unsqueeze(1)).float()
        self_mask = torch.eye(N, device=feats.device)
        pos_mask = pos_mask * (1 - self_mask)

        # Numerically stable log-softmax over all non-self entries
        sim_max = sim.detach().max(dim=1, keepdim=True).values
        exp_sim = torch.exp(sim - sim_max) * (1 - self_mask)
        log_prob = (sim - sim_max) - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Average log-probability over positive pairs per anchor
        pos_count = pos_mask.sum(dim=1)
        valid = pos_count > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=post_inhib.device)

        loss = -(pos_mask * log_prob).sum(dim=1)[valid] / pos_count[valid]
        return loss.mean()

    def _compute_cpc_loss(
        self, refined, raw_levels, temperature=0.1, max_frames=512,
    ):
        """
        Inter-level contrastive predictive coding loss.

        For each adjacent pair (b, b+1), refined[b+1] generates a prediction
        of raw_levels[b] via a learned projection head. InfoNCE contrastive
        loss encourages the prediction to be closer to the true target frame
        than to other time-step frames (negative samples).

        L_b = -log( exp(sim(pred_t, target_t)/tau) / sum_k exp(sim(pred_t, target_k)/tau) )

        This is the neural analogue of predictive coding: higher cortical
        areas predict the activity of lower areas, and the prediction error
        drives learning.

        Args:
            refined: list of 5 tensors, each (B, T, D) — refined block outputs
            raw_levels: list of 5 tensors, each (B, T, D) — raw block outputs (o)
            temperature: InfoNCE temperature
            max_frames: subsample frames for memory efficiency
        Returns:
            scalar loss averaged over all pairs
        """
        total_loss = torch.tensor(0.0, device=refined[0].device)
        num_pairs = 0

        # 4 pairs: refined[1]→raw[0], refined[2]→raw[1], refined[3]→raw[2], refined[4]→raw[3]
        for b in range(len(refined) - 1):
            proj = self.cpc_projections[b]
            pred = proj(refined[b + 1])      # (B, T, D) — prediction of raw[b]
            target = raw_levels[b].detach()   # (B, T, D) — raw target (no grad)

            B, T, D = pred.shape

            # Flatten batch and time
            pred_flat = pred.reshape(B * T, D)
            target_flat = target.reshape(B * T, D)

            # Subsample for memory efficiency (N x N similarity matrix)
            N = min(pred_flat.shape[0], max_frames)
            if pred_flat.shape[0] > N:
                idx = torch.randperm(pred_flat.shape[0], device=pred.device)[:N]
                pred_flat = pred_flat[idx]
                target_flat = target_flat[idx]

            # L2-normalize for cosine similarity
            pred_norm = F.normalize(pred_flat, dim=-1)
            target_norm = F.normalize(target_flat, dim=-1)

            # Similarity matrix: (N, N) — pred[i] vs target[j]
            sim = pred_norm @ target_norm.T / temperature  # (N, N)

            # InfoNCE: each row's diagonal is the positive, rest are negatives
            labels = torch.arange(N, device=sim.device)
            pair_loss = F.cross_entropy(sim, labels)

            total_loss = total_loss + pair_loss
            num_pairs += 1

        return total_loss / max(num_pairs, 1)

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
        
        novel_module_names = {"refine_layers", "inhibition", "ctc_head", "cpc_projections"}
        
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
            if "refine_layers" in name or "inhibition" in name or "ctc_head" in name or "cpc_projections" in name:
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
