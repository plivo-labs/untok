"""Training callback for adding Indic languages to an Untok checkpoint (Lightning)."""
import lightning.pytorch as pl
import torch


class NewSliceLR(pl.Callback):
    """Lightning callback: new-language rows learn faster than the shared layers they sit in.

    NeMo sets learning rates per module, not for a few rows of one layer. The output rows and biases, decoder
    embedding rows, prompt columns and dormant output-tag rows of the trained new languages are mirrored in a small
    side AdamW: the main optimizer sees a zero gradient there, and the side optimizer applies its own rates (linear
    warmup, hold, linear decay to `final_ratio` from `decay_start` of `max_steps`). Before training it refuses any
    listed language whose initialised rows were not calibrated (`untok calibrate`), reading the metadata from the init
    checkpoint.
    """

    def __init__(self, init_checkpoint, languages, lr_rows=3e-3, lr_embedding=1e-3, lr_prompt=1e-3,
                 warmup_steps=100, max_steps=None, decay_start=0.7, final_ratio=0.1, clip=0.5):
        self.init_checkpoint, self.languages = str(init_checkpoint), list(languages)
        self.lrs = {"rows": lr_rows, "embedding": lr_embedding, "prompt": lr_prompt}
        self.warmup_steps, self.max_steps = warmup_steps, max_steps
        self.decay_start, self.final_ratio, self.clip = decay_start, final_ratio, clip
        self.optimizer, self._pending = None, None

    def setup(self, trainer, pl_module, stage):
        from omegaconf import open_dict
        from .language_init import read_language_metadata, require_calibrated

        metadata = read_language_metadata(self.init_checkpoint)
        require_calibrated(metadata, self.languages)
        self.metadata = metadata
        cfg = pl_module.cfg  # keep the record in checkpoints saved from this run
        with open_dict(cfg):
            cfg.untok_languages = metadata
        pl_module.cfg = cfg

    def on_fit_start(self, trainer, pl_module):
        languages = [self.metadata["languages"][locale] for locale in self.languages]
        scripts = sorted({l["script"] for l in languages if l["rows"] == "initialized"})
        tags = self.metadata.get("tags") or {}
        tag_rows = {tags[l]["id"] for l in self.languages if l in tags and tags[l]["init"] != "kept_trained"}
        rows = sorted({r for s in scripts for r in self.metadata["scripts"][s]["rows"]} | tag_rows)
        enc_hidden = int(pl_module.cfg.model_defaults.enc_hidden)
        columns = sorted({enc_hidden + l["prompt_slot"] for l in languages if l["prompt_init"] == "auto_copy"})
        head = pl_module.joint.joint_net[-1]
        embedding = pl_module.decoder.prediction["embed"]
        prompt = pl_module.prompt_kernel[0]
        device = head.weight.device
        index = lambda values: torch.tensor(values, dtype=torch.long, device=device)
        self.slices = []
        if rows:
            self.slices += [(head.weight, 0, index(rows), "rows"), (head.bias, 0, index(rows), "rows"),
                            (embedding.weight, 0, index(rows), "embedding")]
        if columns:
            self.slices.append((prompt.weight, 1, index(columns), "prompt"))
        if not self.slices:
            return
        self.side = [torch.nn.Parameter(t.detach().index_select(d, i).float().clone()) for t, d, i, _ in self.slices]
        self.optimizer = torch.optim.AdamW([{"params": [p], "lr": self.lrs[kind]} for p, (_, _, _, kind)
                                            in zip(self.side, self.slices)], betas=(0.9, 0.98), eps=1e-8, weight_decay=0.0)
        total = self.max_steps or trainer.max_steps
        if not total or total < 1:
            raise ValueError("NewSliceLR needs max_steps (or trainer.max_steps)")
        decay_from = int(total * self.decay_start)

        def factor(step):
            ramp = min(1.0, (step + 1) / max(1, self.warmup_steps))
            if step < decay_from:
                return ramp
            return ramp * max(self.final_ratio, 1 - (1 - self.final_ratio) * (step - decay_from) / max(1, total - decay_from))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, factor)
        if self._pending:
            self.optimizer.load_state_dict(self._pending["optimizer"])
            self.scheduler.load_state_dict(self._pending["scheduler"])
            for parameter, value in zip(self.side, self._pending["side"]):
                parameter.data.copy_(value.to(parameter.device))
        print(f"[untok.NewSliceLR] {len(rows)} new rows ({', '.join(scripts)}; {len(tag_rows)} tags), "
              f"{len(columns)} prompt columns; "
              f"learning rates {self.lrs}", flush=True)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if self.optimizer is None:
            return
        with torch.no_grad():
            for parameter, (tensor, dim, index, _) in zip(self.side, self.slices):
                grad = tensor.grad
                parameter.grad = None if grad is None else grad.index_select(dim, index).float().clone()
                if grad is not None:
                    grad.index_fill_(dim, index, 0.0)
            if self.clip:
                torch.nn.utils.clip_grad_norm_(self.side, self.clip)
            self.optimizer.step()
            self.scheduler.step()
            for parameter, (tensor, dim, index, _) in zip(self.side, self.slices):
                tensor.data.index_copy_(dim, index, parameter.data.to(tensor.dtype))

    def state_dict(self):
        if self.optimizer is None:
            return {}
        return {"optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                "side": [p.detach().cpu() for p in self.side]}

    def load_state_dict(self, state):
        self._pending = state or None
