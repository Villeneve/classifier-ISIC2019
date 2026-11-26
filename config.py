"""Configuração central do projeto."""
from dataclasses import dataclass, field
import torch

# Modelos executados por padrão (sem --models nem --all).
DEFAULT_MODELS = [
    "resnet50",
    "densenet121",
    "efficientnet_b0",
    "efficientnet_b3",
    "mobilenet_v3_large",
    "convnext_tiny",
    "vit_b_16",
]

# Todos os modelos aceitos por --models (e executados por --all), em ordem ascendente de custo
# (GFLOPs na resolução usada pelo projeto; ver README).
AVAILABLE_MODELS = [
    "shufflenet_v2_x1_0",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "regnet_y_400mf",
    "googlenet",
    "regnet_y_1_6gf",
    "resnet18",
    "efficientnet_b3",
    "densenet121",
    "resnet34",
    "resnet50",
    "resnext50_32x4d",
    "convnext_tiny",
    "swin_t",
    "maxvit_t",
    "inception_v3",
    "swin_v2_t",
    "vgg11", "vgg11_bn",
    "efficientnet_v2_s",
    "convnext_small",
    "vgg13", "vgg13_bn",
    "vgg16", "vgg16_bn",
    "vit_b_16",
    "vgg19", "vgg19_bn",
]

# Resolução de entrada por modelo (padrão 224). ViT-B/16 e MaxViT-T exigem 224.
INPUT_SIZE = {"efficientnet_b3": 300, "inception_v3": 299, "swin_v2_t": 256, "efficientnet_v2_s": 384}
DEFAULT_SIZE = 224

# Batch size por modelo (padrão 64) — ajustado para 16 GB de VRAM com bf16.
BATCH_SIZE = {"efficientnet_b3": 32, "convnext_tiny": 48, "swin_t": 48, "densenet121": 48,
              "resnext50_32x4d": 48, "maxvit_t": 32, "swin_v2_t": 32, "convnext_small": 32,
              "efficientnet_v2_s": 24,
              **{f"vgg{n}{s}": 32 for n in (11, 13, 16, 19) for s in ("", "_bn")}}  # VGG usa muita VRAM
DEFAULT_BATCH = 64


@dataclass
class Config:
    data_root: str = "ISIC-2019"        # pasta (ou link simbólico) com train/ e test/ originais
    gpu: int = 1                        # índice da GPU (1 = RTX 5060 Ti; ordem PCI, como no nvidia-smi)
    out_dir: str = "outputs"
    models: list = field(default_factory=lambda: list(DEFAULT_MODELS))
    pretrained: bool = True             # pesos ImageNet (requer internet na 1ª vez)
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 6                   # early stopping (épocas sem melhora do AUC de validação)
    freeze_epochs: int = 0              # épocas iniciais só treinando a cabeça
    num_workers: int = 2                # CPU fraca: poucos workers (só lê o cache uint8, sem decode)
    cache_ram: bool = False             # carrega o cache .npy inteiro na RAM (~15 GB) em vez de mmap
    cache_dir: str = "cache"            # onde ficam os caches uint8 das imagens
    cache_size: int = 450               # lado (px) do cache quadrado
    rebuild_cache: bool = False         # força refazer o cache
    ak: str = "drop"                    # AK: drop | benign | malignant
    subset: int = 0                     # >0: usa só N amostras por split (smoke test)
    batch_size: int = 0                 # >0: sobrescreve BATCH_SIZE
    img_size: int = 0                   # >0: sobrescreve INPUT_SIZE
    compile: bool = False               # torch.compile
    seed: int = 42
    val_frac: float = 0.15              # fração do treino separada para validação
    clear: bool = False                 # apaga o conteúdo de out_dir antes de treinar

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
