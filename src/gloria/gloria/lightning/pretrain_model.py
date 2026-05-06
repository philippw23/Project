import torch

from .. import builder
from pytorch_lightning.core import LightningModule


class PretrainModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters(self.cfg)
        self.gloria = builder.build_gloria_model(cfg)
        self.lr = cfg.lightning.trainer.lr
        self.dm = None

        # Load pretrained CheXpert checkpoint BEFORE injecting LoRA
        if cfg.model.get("pretrain_ckpt"):
            ckpt = torch.load(cfg.model.pretrain_ckpt, map_location="cpu")
            state = {
                k.replace("gloria.", "", 1): v
                for k, v in ckpt["state_dict"].items()
                if k.startswith("gloria.")
            }
            missing, unexpected = self.gloria.load_state_dict(state, strict=False)
            print(
                f"Loaded CheXpert checkpoint from {cfg.model.pretrain_ckpt}\n"
                f"  missing keys   : {len(missing)}\n"
                f"  unexpected keys: {len(unexpected)}"
            )

        # Inject LoRA into image encoder AFTER loading pretrained weights
        if cfg.model.vision.get("lora"):
            from ..models.lora import inject_lora_gloria, count_trainable_params
            inject_lora_gloria(
                self.gloria.img_encoder,
                lora_layers=cfg.model.vision.lora.lora_layers,
                r=cfg.model.vision.lora.r,
                alpha=cfg.model.vision.lora.alpha,
            )
            n = count_trainable_params(self.gloria.img_encoder)
            print(f"LoRA injected: {n:,} trainable params in img_encoder")

    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.lr, self.gloria)
        scheduler = builder.build_scheduler(self.cfg, optimizer, self.dm)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        loss, attn_maps, sents = self.shared_step(batch, "train")

        # get attention map image
        if self.cfg.train.update_interval is not None:
            if batch_idx % self.cfg.train.update_interval == 0:
                imgs = batch["imgs"].cpu()
                self.gloria.plot_attn_maps(
                    attn_maps, imgs, sents, self.current_epoch, batch_idx
                )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self.shared_step(batch, "val")
        return loss

    def shared_step(self, batch, split):
        """Similar to traning step"""

        img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents = self.gloria(batch)
        loss, attn_maps = self.gloria.calc_loss(
            img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents
        )

        # log training progress
        log_iter_loss = True if split == "train" else False
        self.log(
            f"{split}_loss",
            loss,
            on_epoch=True,
            on_step=log_iter_loss,
            logger=True,
            prog_bar=True,
        )

        return loss, attn_maps, sents
