from dataclasses import dataclass

from ubigraph_core.config import Config


@dataclass
class STConfig(Config):
    synthetic_nodes: int = 96  # synchronized episodes, not parking lots
    epochs: int = 100
    patience: int = 25
    learning_rate: float = 0.002
    batch_size: int = 256
    validation_interval: int = 5
    condense_steps: int = 100
    real_batch: int = 12
    feature_lr: float = 0.002
    adjacency_lr: float = 0.002
    theta_lr: float = 0.005
    magnitude_weight: float = 0.1
    anchor_weight: float = 0.1
    shape_weight: float = 0.1
    retrieval_weight: float = 0.5
    adversarial_epsilon: float = 0.01
    adversarial_lr: float = 0.02
    adversarial_steps: int = 2
    trust_radius: float = 0.05
    edge_logit_scale: float = 0.5
    correction_scale: float = 0.1
    delta_scale: float = 0.05
    attention_prior: float = 0.5
    adaptation_epochs: int = 80
    adaptation_lr: float = 0.003
    adaptation_fraction: float = 0.75
    seeds: tuple[int, ...] = (7, 17, 27)
    folds: bool = False
    ablations: bool = False

    methods: tuple[str, ...] = ("groc",)
    validation_layout: str | None = None
    test_layout: str | None = None

    def validate(self):
        super().validate()
        if self.revin:
            raise ValueError("V3 uses capacity normalization, not RevIN")
        if self.synthetic_nodes <= self.neighbors + 4:
            raise ValueError("Need context and supervised synchronized episodes")
        for key in ("batch_size", "validation_interval", "adaptation_epochs", "trust_radius", "delta_scale"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if not .5 <= self.adaptation_fraction < 1:
            raise ValueError("adaptation_fraction must be in [.5,1)")
        if int(self.fewshot * self.adaptation_fraction) <= self.neighbors:
            raise ValueError("Few-shot fit partition needs context and supervised queries")
        for key in ("shape_weight", "retrieval_weight", "attention_prior", "edge_logit_scale"):
            if getattr(self, key) < 0:
                raise ValueError(f"{key} must be nonnegative")
