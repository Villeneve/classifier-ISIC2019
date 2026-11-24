"""Configuração central do projeto."""
from dataclasses import dataclass, field
import torch

ALL_MODELS = [
    "resnet50",
    "densenet121",
    "efficientnet_b0",
    "efficientnet_b3",
    "mobilenet_v3_large",
    "convnext_tiny",
    "vit_b_16",
]

# Resolução de entrada por modelo (padrão 224). ViT-B/16 exige 224.
INPUT_SIZE = {"efficientnet_b3": 300}
DEFAULT_SIZE = 224

# Batch size por modelo (padrão 64) — ajustado para 16 GB de VRAM com bf16.
BATCH_SIZE = {"efficientnet_b3": 32, "convnext_tiny": 48, "densenet121": 48}
DEFAULT_BATCH = 64


@dataclass
class Config:
    data_root: str = "pth"              # pasta (ou link simbólico) com {train,test}_{images,labels}.pth
    gpu: int = 1                        # índice da GPU (1 = RTX 5060 Ti; ordem PCI, como no nvidia-smi)
    out_dir: str = "outputs"
    models: list = field(default_factory=lambda: list(ALL_MODELS))
    pretrained: bool = True             # pesos ImageNet (requer internet na 1ª vez)
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 6                   # early stopping (épocas sem melhora do AUC de validação)
    freeze_epochs: int = 0              # épocas iniciais só treinando a cabeça
    num_workers: int = 2                # CPU fraca: poucos workers (só lê .pt, sem decode)
    cache_ram: bool = False             # carrega os .pth inteiros na RAM em vez de mmap
    subset: int = 0                     # >0: usa só N amostras por split (smoke test)
    batch_size: int = 0                 # >0: sobrescreve BATCH_SIZE
    img_size: int = 0                   # >0: sobrescreve INPUT_SIZE
    compile: bool = False               # torch.compile
    seed: int = 42
    val_frac: float = 0.15              # fração do treino separada para validação

    @property
    def device(self):
        if not torch.cuda.is_available():
            return torch.device("cpu")
        if not 0 <= self.gpu < torch.cuda.device_count():
            raise ValueError(f"--gpu {self.gpu} inválida: há {torch.cuda.device_count()} GPU(s) visíveis")
        return torch.device(f"cuda:{self.gpu}")

    def size_for(self, name):
        return self.img_size or INPUT_SIZE.get(name, DEFAULT_SIZE)

    def batch_for(self, name):
        return self.batch_size or BATCH_SIZE.get(name, DEFAULT_BATCH)
