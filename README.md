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
O projeto usa os arquivos originais do ISIC-2019 em `ISIC-2019/` (link simbólico na raiz do projeto, ignorado pelo git):

```
ISIC-2019/train/ISIC_2019_Training_Input/*.jpg   + ISIC_2019_Training_GroundTruth.csv + ISIC_2019_Training_Metadata.csv
ISIC-2019/test/ISIC_2019_Test_Input/*.jpg        + ISIC_2019_Test_GroundTruth.csv
```
```bash
ln -s /caminho/real/da/pasta/ISIC-2019 ISIC-2019
```
(A pasta `pth/` e o `pth.7z` não são usados.)

**Tratamento dos dados**
- **Cache (1ª execução):** os JPEGs têm tamanhos variados. Cada um é decodificado uma vez (nvjpeg na GPU, por
  causa da CPU fraca), redimensionado (lado menor = 450, antialias) com recorte central quadrado e salvo em
  `cache/{train,test}_450.npy` (uint8 `[N,3,450,450]`, ~15 GB + ~5 GB, lido com mmap). As próximas execuções
  só leem o cache. `--rebuild-cache` refaz.
- **Rótulo binário:** maligno = MEL, BCC, SCC; benigno = NV, BKL, DF, VASC. **AK** (queratose actínica, pré-maligna)
  é controlada por `--ak {benign,malignant,drop}` (padrão `benign`). Linhas sem classe/UNK são descartadas.
- **Validação:** 15% do `train/`, separada **por lesão** (`lesion_id` do Metadata) para não vazar imagens da mesma
  lesão entre treino e validação, estratificada por classe, seed fixa. O `test/` é usado só no teste final.
  Os ids de cada split ficam em `outputs/split.json`.

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
`--img-size`, `--num-workers`, `--ak`, `--cache-dir`, `--rebuild-cache`, `--cache-ram` (cache inteiro na RAM, ~15 GB), `--compile`, `--no-pretrained`, `--seed`.

## Estrutura
```
main.py              orquestra: summary, treino, teste e comparação por modelo
config.py            hiperparâmetros, modelos, tamanhos de entrada e batch por modelo
data/dataset.py      CSVs, cache uint8 dos JPEGs, rótulo binário, split por lesão, DataLoaders, pos_weight
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
