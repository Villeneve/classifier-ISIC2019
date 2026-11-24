# Classificador ISIC-2019 (benigno × maligno)

Projeto em PyTorch para comparar arquiteturas (CNNs e ViT) na classificação binária do ISIC-2019.
Classe positiva = **maligno** (1); negativa = **benigno** (0).

## Ambiente-alvo
RTX 5060 Ti 16 GB, Ubuntu 24.04, AMD FX-8320E, 32 GB RAM, Python 3.10.

```bash
# A RTX 5060 Ti (Blackwell, sm_120) exige PyTorch >= 2.7 com CUDA 12.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install torchinfo tqdm numpy matplotlib
```

## Dataset
O projeto lê a pasta `pth/` (na raiz do projeto, normalmente um link simbólico; ignorada pelo git):

```
pth/train_images.pth  [N,3,450,450] uint8     pth/train_labels.pth  [N] ou [N,1]
pth/test_images.pth   [M,3,450,450] uint8     pth/test_labels.pth   [M] ou [M,1]
```
Rótulos binários (0 benigno, 1 maligno). Não há validação pronta: ela é separada do treino
(15%, estratificada, seed fixa; índices em `outputs/split.json`). Os `.pth` são abertos com mmap.

```bash
ln -s /caminho/real/da/pasta/pth pth
```

## GPU
Padrão: GPU **1** (RTX 5060 Ti). Use `--gpu 0` para a outra. O índice segue a ordem do `nvidia-smi`
(`CUDA_DEVICE_ORDER=PCI_BUS_ID` é definido em `main.py`) e o nome da GPU é impresso ao iniciar.

## Uso
```bash
# smoke test
python main.py --models resnet50 --epochs 1 --subset 200

# todas as arquiteturas
python main.py
```

Opções úteis: `--gpu`, `--data-root`, `--models`, `--epochs`, `--lr`, `--patience`, `--freeze-epochs`, `--batch-size`,
`--img-size`, `--num-workers`, `--cache-ram` (carrega os .pth na RAM), `--compile`, `--no-pretrained`, `--seed`.

## Estrutura
```
main.py              orquestra: summary, treino, teste e comparação por modelo
config.py            hiperparâmetros, modelos, tamanhos de entrada e batch por modelo
data/dataset.py      leitura dos .pth, split de validação, DataLoaders, pos_weight
data/augment.py      pré-processamento e augmentation em batch na GPU
models/factory.py    modelos torchvision com cabeça de 1 logit
engine/train.py      loop de treino/validação, early stopping, checkpoint
engine/metrics.py    AUC, ROC, sensibilidade, especificidade, F1... (numpy)
utils/plots.py       curvas, ROC, matriz de confusão, comparação
utils/seed.py        reprodutibilidade
```

## Arquiteturas
`resnet50`, `densenet121`, `efficientnet_b0`, `efficientnet_b3` (300 px), `mobilenet_v3_large`,
`convnext_tiny`, `vit_b_16`. Pesos ImageNet (exigem internet na primeira execução).

## Decisões de projeto
- **Saída de 1 logit** com `BCEWithLogitsLoss(pos_weight = n_benigno / n_maligno)` para o desbalanceamento.
- **Augmentation na GPU**: a CPU do alvo é fraca; o DataLoader só entrega tensores uint8 e
  rotação, flip, zoom, brilho, contraste e saturação são aplicados por amostra no batch em `cuda`.
- **Velocidade/memória**: bf16, `channels_last`, `cudnn.benchmark`, `torch.compile` opcional.
- **Treino**: AdamW + CosineAnnealing, early stopping pelo AUC de validação, melhor checkpoint restaurado.
- **Limiar**: escolhido na validação (índice de Youden) e aplicado no teste; métricas em 0,5 também são salvas.

### Por que `torch.autocast(..., enabled=device.type == "cuda")`?
O `autocast` executa as operações em precisão reduzida (aqui bfloat16) para ganhar velocidade e
economizar VRAM na GPU. O argumento `enabled` permite **usar o mesmo código em qualquer dispositivo**
sem `if/else` espalhados:

- Em `cuda`, `enabled=True` → bf16 ativo.
- Em `cpu` (ex.: teste de sintaxe/smoke test em máquina sem GPU), `enabled=False` → o `autocast` vira
  um no-op e tudo roda em float32. Sem isso, o autocast de CPU com bf16 funcionaria, mas costuma ser
  mais lento ou instável que fp32 puro, além de mudar a numérica entre os dois ambientes.

Como o `device` vem do `Config` (`cuda` se disponível, senão `cpu`), `enabled` faz a precisão mista
acontecer só onde compensa. Note que as losses e probabilidades são calculadas em float32
(`logits.float()`) para estabilidade, e bf16 não precisa de `GradScaler`.

## Saídas (`outputs/`)
```
split.json  summary.json  comparison.png
<modelo>/best.pt  history.json  history.png  metrics.json  roc.png  confusion.png  summary.txt
```
