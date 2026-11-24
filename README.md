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
Um `.pt` por imagem (uint8, RGB, 450×450). A estrutura de pastas é lida por `_scan()` em `data/dataset.py`:

```
root/{train,val,test}/{benign,malignant}/*.pt     # splits prontos
root/{benign,malignant}/*.pt                      # split estratificado 70/15/15 automático
```
Pastas de classe aceitas: `benign`/`benigno`/`0` e `malignant`/`maligno`/`1`. Se o seu layout for
diferente, só `_scan()` precisa mudar. O split usado é salvo em `outputs/split.json`.

## Uso
```bash
# smoke test
python main.py --data-root /caminho/dataset --models resnet50 --epochs 1 --subset 200

# todas as arquiteturas
python main.py --data-root /caminho/dataset
```

Opções úteis: `--models`, `--epochs`, `--lr`, `--patience`, `--freeze-epochs`, `--batch-size`,
`--img-size`, `--num-workers`, `--cache-ram` (~15 GB), `--compile`, `--no-pretrained`, `--seed`.

## Estrutura
```
main.py              orquestra: summary, treino, teste e comparação por modelo
config.py            hiperparâmetros, modelos, tamanhos de entrada e batch por modelo
data/dataset.py      _scan, splits, Dataset de .pt, DataLoaders, pos_weight
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
