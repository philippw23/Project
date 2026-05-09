# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from functools import partial
import logging

import torch
from torch import nn

from chexfound.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss
from chexfound.models import build_model_from_cfg
from chexfound.layers import DINOHead
from chexfound.utils.utils import has_batchnorms
from chexfound.utils.param_groups import get_params_groups_with_decay, fuse_params_groups
from chexfound.fsdp import get_fsdp_wrapper, ShardedGradScaler, get_fsdp_modules, reshard_fsdp_model

from chexfound.models.vision_transformer import BlockChunk


try:
    from xformers.ops import fmha
except ImportError:
    raise AssertionError("xFormers is required for training")


logger = logging.getLogger("chexfound")


class SSLMetaArch(nn.Module):
    """Joint DINO + iBOT self-supervised learning architecture.

    Maintains a student network that is trained via gradient descent and an
    exponential-moving-average (EMA) teacher network that provides stable
    learning targets. The teacher is never updated by gradients — only by the
    EMA rule applied in ``update_teacher``.

    Three loss terms are combined:
    - **DINO** (``DINOLoss``): knowledge-distillation loss on CLS tokens.
      Student local- and global-crop CLS tokens are matched against teacher
      global-crop CLS tokens, encouraging view-invariant representations.
    - **iBOT** (``iBOTPatchLoss``): masked-image-modelling loss on patch tokens.
      Masked patch tokens from the student are matched against the corresponding
      unmasked patch tokens from the teacher.
    - **KoLeo** (``KoLeoLoss``): entropy-maximisation regulariser that spreads
      student CLS embeddings uniformly across the representation space,
      preventing collapse.

    Args:
        cfg: OmegaConf config object with sections ``student``, ``teacher``,
            ``dino``, ``ibot``, ``crops``, ``train``, ``optim``, and
            ``compute_precision``.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # fp16_scaler is only used with FSDP multi-GPU training; disabled for
        # single-GPU runs by setting compute_precision.grad_scaler = False.
        self.fp16_scaler = ShardedGradScaler() if cfg.compute_precision.grad_scaler else None

        student_model_dict = dict()
        teacher_model_dict = dict()

        # Build backbone (ViT) for both student and teacher from the same config.
        # embed_dim is the backbone output dimension (e.g. 1024 for ViT-L).
        student_backbone, teacher_backbone, embed_dim = build_model_from_cfg(cfg)
        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone
        logger.info(f"OPTIONS -- architecture : embed_dim: {embed_dim}")

        if cfg.student.pretrained_weights:
            chkpt = torch.load(cfg.student.pretrained_weights)
            logger.info(f"OPTIONS -- pretrained weights: loading from {cfg.student.pretrained_weights}")
            student_backbone.load_state_dict(chkpt["model"], strict=False)

        self.embed_dim = embed_dim
        self.dino_out_dim = cfg.dino.head_n_prototypes  # number of prototype vectors

        # Flags that control which loss terms are active.
        self.do_dino = cfg.dino.loss_weight > 0
        self.do_koleo = cfg.dino.koleo_loss_weight > 0
        self.do_ibot = cfg.ibot.loss_weight > 0
        self.ibot_separate_head = cfg.ibot.separate_head  # True = iBOT uses its own projection head

        logger.info("OPTIONS -- DINO")
        if self.do_dino:
            logger.info(f"OPTIONS -- DINO -- loss_weight: {cfg.dino.loss_weight}")
            logger.info(f"OPTIONS -- DINO -- head_n_prototypes: {cfg.dino.head_n_prototypes}")
            logger.info(f"OPTIONS -- DINO -- head_bottleneck_dim: {cfg.dino.head_bottleneck_dim}")
            logger.info(f"OPTIONS -- DINO -- head_hidden_dim: {cfg.dino.head_hidden_dim}")
            self.dino_loss_weight = cfg.dino.loss_weight

            # Factory for DINOHead: projects embed_dim → bottleneck → n_prototypes.
            # Called twice below (student + teacher) to produce independent instances.
            dino_head = partial(
                DINOHead,
                in_dim=embed_dim,
                out_dim=cfg.dino.head_n_prototypes,
                hidden_dim=cfg.dino.head_hidden_dim,
                bottleneck_dim=cfg.dino.head_bottleneck_dim,
                nlayers=cfg.dino.head_nlayers,
            )
            self.dino_loss = DINOLoss(self.dino_out_dim)
            if self.do_koleo:
                logger.info("OPTIONS -- DINO -- applying KOLEO regularization")
                self.koleo_loss = KoLeoLoss()
        else:
            logger.info("OPTIONS -- DINO -- not using DINO")

        if self.do_dino or self.do_ibot:
            student_model_dict["dino_head"] = dino_head()
            teacher_model_dict["dino_head"] = dino_head()

        logger.info("OPTIONS -- IBOT")
        logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_ratio_tuple: {cfg.ibot.mask_ratio_min_max}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_sample_probability: {cfg.ibot.mask_sample_probability}")
        if self.do_ibot:
            self.ibot_loss_weight = cfg.ibot.loss_weight
            assert max(cfg.ibot.mask_ratio_min_max) > 0, "please provide a positive mask ratio tuple for ibot"
            assert cfg.ibot.mask_sample_probability > 0, "please provide a positive mask probability for ibot"

            # Output dim: use iBOT-specific prototypes if a separate head is configured,
            # otherwise share the DINO prototype space.
            self.ibot_out_dim = cfg.ibot.head_n_prototypes if self.ibot_separate_head else cfg.dino.head_n_prototypes
            self.ibot_patch_loss = iBOTPatchLoss(self.ibot_out_dim)
            if self.ibot_separate_head:
                logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
                logger.info(f"OPTIONS -- IBOT -- head_n_prototypes: {cfg.ibot.head_n_prototypes}")
                logger.info(f"OPTIONS -- IBOT -- head_bottleneck_dim: {cfg.ibot.head_bottleneck_dim}")
                logger.info(f"OPTIONS -- IBOT -- head_hidden_dim: {cfg.ibot.head_hidden_dim}")
                ibot_head = partial(
                    DINOHead,
                    in_dim=embed_dim,
                    out_dim=cfg.ibot.head_n_prototypes,
                    hidden_dim=cfg.ibot.head_hidden_dim,
                    bottleneck_dim=cfg.ibot.head_bottleneck_dim,
                    nlayers=cfg.ibot.head_nlayers,
                )
                student_model_dict["ibot_head"] = ibot_head()
                teacher_model_dict["ibot_head"] = ibot_head()
            else:
                logger.info("OPTIONS -- IBOT -- head shared with DINO")

        # Flag checked by fsdp_synchronize_streams(); disabled for single-GPU runs.
        self.need_to_synchronize_fsdp_streams = True

        # Wrap submodels in ModuleDicts so FSDP can shard them independently.
        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        # Teacher is never updated by gradients — only by EMA in update_teacher().
        for p in self.teacher.parameters():
            p.requires_grad = False
        logger.info(f"Student and Teacher are built: they are both {cfg.student.arch} network.")

    def forward(self, inputs):
        raise NotImplementedError

    def backprop_loss(self, loss):
        """Run backward pass, scaling with fp16 scaler when active."""
        if self.fp16_scaler is not None:
            self.fp16_scaler.scale(loss).backward()
        else:
            loss.backward()

    def forward_backward(self, images, teacher_temp):
        """Run one full forward + backward pass and return per-term loss values.

        Flow:
        1. Teacher forward (no grad): process both global crops → get CLS tokens
           and patch tokens → project through head(s) → apply centering/
           Sinkhorn-Knopp normalisation to produce soft targets.
        2. Student forward (with grad): process global crops (with mask tokens at
           masked positions) and local crops → collect CLS and masked patch tokens
           → project through head(s) in a single fused xFormers call.
        3. Compute DINO loss (CLS tokens), iBOT loss (masked patch tokens), and
           optionally KoLeo regularisation.
        4. Backpropagate the weighted sum of all active loss terms.

        Args:
            images: Dict produced by ``collate_data_and_cast`` containing stacked
                crop tensors and patch mask metadata.
            teacher_temp: Current teacher softmax temperature (annealed during
                warmup epochs).

        Returns:
            Dict mapping loss name → scalar tensor for logging. Keys present
            depend on which losses are active: ``dino_local_crops_loss``,
            ``dino_global_crops_loss``, ``ibot_loss``, ``koleo_loss``.
        """
        n_global_crops = 2
        assert n_global_crops == 2
        n_local_crops = self.cfg.crops.local_crops_number

        # Move all batch tensors to GPU.
        global_crops = images["collated_global_crops"].cuda(non_blocking=True)
        local_crops = images["collated_local_crops"].cuda(non_blocking=True)

        masks = images["collated_masks"].cuda(non_blocking=True)
        mask_indices_list = images["mask_indices_list"].cuda(non_blocking=True)
        n_masked_patches_tensor = images["n_masked_patches"].cuda(non_blocking=True)
        n_masked_patches = mask_indices_list.shape[0]  # total masked patches across batch
        upperbound = images["upperbound"]              # pre-allocated buffer size
        masks_weight = images["masks_weight"].cuda(non_blocking=True)

        # Number of (student, teacher) crop pairs used in each loss term.
        n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
        n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops

        do_dino = self.do_dino
        do_ibot = self.do_ibot

        # iBOT loss is averaged over the two global crops.
        ibot_loss_scale = 1.0 / n_global_crops

        @torch.no_grad()
        def get_teacher_output():
            """Forward teacher on unmasked global crops and produce soft targets.

            Returns:
                teacher_dino_softmaxed_centered_list: Soft DINO targets for each
                    global crop view, shape (n_global_crops, B, n_prototypes).
                masked_teacher_ibot_softmaxed_centered: Soft iBOT targets for
                    all masked patch positions, shape (n_masked_patches, n_prototypes).
                    None when iBOT is disabled.
            """
            x, n_global_crops_teacher = global_crops, n_global_crops

            # Full forward through teacher backbone — returns CLS and patch tokens.
            teacher_backbone_output_dict = self.teacher.backbone(x, is_training=True)
            teacher_cls_tokens = teacher_backbone_output_dict["x_norm_clstoken"]

            # Split the batch back into the two global crop views, then swap order
            # so each view acts as target for the other (cross-view prediction).
            teacher_cls_tokens = teacher_cls_tokens.chunk(n_global_crops_teacher)
            teacher_cls_tokens = torch.cat((teacher_cls_tokens[1], teacher_cls_tokens[0]))

            ibot_teacher_patch_tokens = teacher_backbone_output_dict["x_norm_patchtokens"]
            _dim = ibot_teacher_patch_tokens.shape[-1]
            n_cls_tokens = teacher_cls_tokens.shape[0]

            if do_ibot and not self.ibot_separate_head:
                # Shared head: run CLS tokens and masked patch tokens through the
                # same DINO head in one call using a pre-allocated buffer.
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound + n_cls_tokens, _dim)
                buffer_tensor_teacher[:n_cls_tokens].copy_(teacher_cls_tokens)
                # Gather only the masked patch positions to avoid processing all 1024 tokens.
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[n_cls_tokens : n_cls_tokens + n_masked_patches],
                )
                tokens_after_head = self.teacher.dino_head(buffer_tensor_teacher)
                teacher_cls_tokens_after_head = tokens_after_head[:n_cls_tokens]
                masked_teacher_patch_tokens_after_head = tokens_after_head[
                    n_cls_tokens : n_cls_tokens + n_masked_patches
                ]
            elif do_ibot and self.ibot_separate_head:
                # Separate heads: route CLS tokens through dino_head and masked
                # patch tokens through ibot_head independently.
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound, _dim)
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[:n_masked_patches],
                )
                teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
                masked_teacher_patch_tokens_after_head = self.teacher.ibot_head(buffer_tensor_teacher)[
                    :n_masked_patches
                ]
            else:
                teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
                masked_teacher_ibot_softmaxed_centered = None

            # Normalise teacher logits into soft probability targets.
            # Two strategies: EMA-based centering or Sinkhorn-Knopp (equipartition).
            if self.cfg.train.centering == "centering":
                teacher_dino_softmaxed_centered_list = self.dino_loss.softmax_center_teacher(
                    teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                ).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])
                self.dino_loss.update_center(teacher_cls_tokens_after_head)
                if do_ibot:
                    masked_teacher_patch_tokens_after_head = masked_teacher_patch_tokens_after_head.unsqueeze(0)
                    masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.softmax_center_teacher(
                        masked_teacher_patch_tokens_after_head[:, :n_masked_patches], teacher_temp=teacher_temp
                    )
                    masked_teacher_ibot_softmaxed_centered = masked_teacher_ibot_softmaxed_centered.squeeze(0)
                    self.ibot_patch_loss.update_center(masked_teacher_patch_tokens_after_head[:n_masked_patches])

            elif self.cfg.train.centering == "sinkhorn_knopp":
                # Sinkhorn-Knopp iteratively normalises rows and columns of the
                # assignment matrix, producing a uniform marginal over prototypes.
                teacher_dino_softmaxed_centered_list = self.dino_loss.sinkhorn_knopp_teacher(
                    teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                ).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])

                if do_ibot:
                    masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
                        masked_teacher_patch_tokens_after_head,
                        teacher_temp=teacher_temp,
                        n_masked_patches_tensor=n_masked_patches_tensor,
                    )
            else:
                raise NotImplementedError

            return teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered

        teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered = get_teacher_output()
        reshard_fsdp_model(self.teacher)

        loss_dict = {}
        loss_accumulator = 0

        # ── Student forward ───────────────────────────────────────────────────
        # Pass both global and local crops through the backbone in one call.
        # Global crops are forwarded with the patch mask so masked positions
        # receive a learnable mask token instead of their real patch embedding.
        # Local crops are forwarded without a mask (masks=None).
        student_global_backbone_output_dict, student_local_backbone_output_dict = self.student.backbone(
            [global_crops, local_crops], masks=[masks, None], is_training=True
        )

        # Collect all CLS token inputs for the DINO head in a list.
        # They will be concatenated and forwarded in one fused xFormers call below.
        inputs_for_student_head_list = []

        student_local_cls_tokens = student_local_backbone_output_dict["x_norm_clstoken"]
        inputs_for_student_head_list.append(student_local_cls_tokens.unsqueeze(0))

        student_global_cls_tokens = student_global_backbone_output_dict["x_norm_clstoken"]
        inputs_for_student_head_list.append(student_global_cls_tokens.unsqueeze(0))

        if do_ibot:
            _dim = student_global_backbone_output_dict["x_norm_clstoken"].shape[-1]
            ibot_student_patch_tokens = student_global_backbone_output_dict["x_norm_patchtokens"]

            # Gather only the masked patch tokens using the flat index list —
            # avoids passing all 1024 patch tokens through the iBOT head.
            buffer_tensor_patch_tokens = ibot_student_patch_tokens.new_zeros(upperbound, _dim)
            buffer_tensor_patch_tokens[:n_masked_patches].copy_(
                torch.index_select(ibot_student_patch_tokens.flatten(0, 1), dim=0, index=mask_indices_list)
            )
            if not self.ibot_separate_head:
                # Shared head: include masked patch tokens in the fused head call.
                inputs_for_student_head_list.append(buffer_tensor_patch_tokens.unsqueeze(0))
            else:
                # Separate head: forward masked patch tokens through ibot_head now.
                student_global_masked_patch_tokens_after_head = self.student.ibot_head(buffer_tensor_patch_tokens)[
                    :n_masked_patches
                ]

        # ── Fused DINO head call (xFormers BlockDiagonalMask) ────────────────
        # Concatenate all inputs into one tensor and forward through the DINO head
        # in a single efficient call, then split outputs back to their original groups.
        _attn_bias, cat_inputs = fmha.BlockDiagonalMask.from_tensor_list(inputs_for_student_head_list)
        outputs_list = _attn_bias.split(self.student.dino_head(cat_inputs))

        student_local_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)
        student_global_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        if do_ibot and not self.ibot_separate_head:
            student_global_masked_patch_tokens_after_head = outputs_list.pop(0).squeeze(0)[:n_masked_patches]

        # ── Loss computation ─────────────────────────────────────────────────

        if n_local_crops > 0:
            # DINO local: each local crop's CLS token vs. both teacher global CLS targets.
            dino_local_crops_loss = self.dino_loss(
                student_output_list=student_local_cls_tokens_after_head.chunk(n_local_crops),
                teacher_out_softmaxed_centered_list=teacher_dino_softmaxed_centered_list,
            ) / (n_global_crops_loss_terms + n_local_crops_loss_terms)

            loss_dict["dino_local_crops_loss"] = dino_local_crops_loss
            loss_accumulator += self.dino_loss_weight * dino_local_crops_loss

        # loss_scales=2 compensates for the cross-view swap applied to teacher tokens.
        loss_scales = 2

        if do_dino:
            # DINO global: each global crop's CLS token vs. the other global crop's teacher target.
            dino_global_crops_loss = (
                self.dino_loss(
                    student_output_list=[student_global_cls_tokens_after_head],
                    teacher_out_softmaxed_centered_list=[
                        teacher_dino_softmaxed_centered_list.flatten(0, 1)
                    ],
                )
                * loss_scales
                / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            )

            loss_dict["dino_global_crops_loss"] = dino_global_crops_loss
            loss_accumulator += self.dino_loss_weight * dino_global_crops_loss

            student_cls_tokens = student_global_cls_tokens

            if self.do_koleo:
                # KoLeo: spread the two global-crop CLS embeddings uniformly in
                # representation space to prevent prototype collapse.
                koleo_loss = self.cfg.dino.koleo_loss_weight * sum(
                    self.koleo_loss(p) for p in student_cls_tokens.chunk(2)
                )
                loss_accumulator += koleo_loss
                loss_dict["koleo_loss"] = koleo_loss / loss_scales

        if do_ibot:
            # iBOT: student masked patch tokens vs. teacher unmasked patch targets.
            ibot_patch_loss = (
                self.ibot_patch_loss.forward_masked(
                    student_global_masked_patch_tokens_after_head,
                    masked_teacher_ibot_softmaxed_centered,
                    student_masks_flat=masks,
                    n_masked_patches=n_masked_patches,
                    masks_weight=masks_weight,
                )
                * loss_scales
                * ibot_loss_scale
            )

            loss_dict["ibot_loss"] = ibot_patch_loss / 2
            loss_accumulator += self.ibot_loss_weight * ibot_patch_loss

        self.backprop_loss(loss_accumulator)
        self.fsdp_synchronize_streams()

        return loss_dict

    def fsdp_synchronize_streams(self):
        """Align CUDA streams across FSDP shards after the first backward pass.

        FSDP uses separate CUDA streams for communication and computation. This
        method synchronises them once so subsequent iterations proceed without
        stalls. No-op for single-GPU runs (need_to_synchronize_fsdp_streams=False).
        """
        if self.need_to_synchronize_fsdp_streams:
            torch.cuda.synchronize()
            self.student.dino_head._streams = (
                self.teacher.dino_head._streams
            ) = self.student.backbone._streams = self.teacher.backbone._streams
            self.need_to_synchronize_fsdp_streams = False

    def update_teacher(self, m):
        """Update teacher parameters via exponential moving average (EMA).

        For each parameter ``p_t`` in the teacher and corresponding ``p_s`` in
        the student:  ``p_t ← m * p_t + (1 - m) * p_s``

        The momentum ``m`` is typically annealed from ~0.994 to 1.0 during
        training, so the teacher updates become increasingly conservative.

        Args:
            m: EMA momentum coefficient (closer to 1 = slower teacher update).
        """
        student_param_list = []
        teacher_param_list = []
        with torch.no_grad():
            for k in self.student.keys():
                for ms, mt in zip(get_fsdp_modules(self.student[k]), get_fsdp_modules(self.teacher[k])):
                    student_param_list += ms.params
                    teacher_param_list += mt.params
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

    def train(self):
        """Set student to train mode while keeping teacher in eval mode.

        The teacher must stay in eval mode to keep its BatchNorm statistics
        frozen and dropout disabled, ensuring stable EMA targets.
        """
        super().train()
        self.teacher.eval()

    def get_maybe_fused_params_for_submodel(self, m):
        """Build fused param groups with layerwise LR decay for one submodel.

        Groups parameters by depth and applies a multiplicative LR decay so
        earlier layers receive a smaller learning rate than later ones.
        Groups are then fused for efficient ``foreach`` optimizer operations.

        Args:
            m: A student submodule (backbone, dino_head, or ibot_head).

        Returns:
            List of parameter group dicts compatible with ``torch.optim.AdamW``,
            each with ``foreach=True`` set for vectorised updates.
        """
        params_groups = get_params_groups_with_decay(
            model=m,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
        )
        fused_params_groups = fuse_params_groups(params_groups)
        logger.info("fusing param groups")
        for g in fused_params_groups:
            g["foreach"] = True
        return fused_params_groups

    def get_params_groups(self):
        """Return all student parameter groups for the optimizer.

        Collects fused param groups from every student submodel (backbone,
        dino_head, and optionally ibot_head).

        Returns:
            List of parameter group dicts ready to pass to ``torch.optim.AdamW``.
        """
        all_params_groups = []
        for m in self.student.values():
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        return all_params_groups

    def prepare_for_distributed_training(self):
        """Wrap student and teacher submodels with FSDP for multi-GPU training.

        Syncs student → teacher weights first, then wraps each submodel with
        the precision and sharding strategy defined in ``compute_precision``.
        ``BlockChunk`` is used as the FSDP wrapping unit so each transformer
        block chunk is sharded independently.
        """
        logger.info("DISTRIBUTED FSDP -- preparing model for distributed training")
        if has_batchnorms(self.student):
            raise NotImplementedError
        for k, v in self.student.items():
            self.teacher[k].load_state_dict(self.student[k].state_dict())
            student_model_cfg = self.cfg.compute_precision.student[k]
            self.student[k] = get_fsdp_wrapper(student_model_cfg, modules_to_wrap={BlockChunk})(self.student[k])
            teacher_model_cfg = self.cfg.compute_precision.teacher[k]
            self.teacher[k] = get_fsdp_wrapper(teacher_model_cfg, modules_to_wrap={BlockChunk})(self.teacher[k])
