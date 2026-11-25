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
# smoke test (pasta própria, para não substituir um resultado real)
python main.py --models resnet50 --epochs 1 --subset 200 --out-dir outputs/smoke

# modelos padrão (★ na tabela de arquiteturas)
python main.py

# todos os modelos disponíveis
python main.py --all

# só alguns
python main.py --models resnet18 swin_t

# apaga os resultados anteriores e treina do zero (combina com --all ou --models)
python main.py --clear --all
```

Opções úteis: `--gpu`, `--data-root`, `--models` ou `--all` (um ou outro), `--clear`, `--out-dir`, `--epochs`, `--lr`, `--patience`, `--freeze-epochs`, `--batch-size`,
`--img-size`, `--num-workers`, `--ak`, `--cache-dir`, `--rebuild-cache`, `--cache-ram` (cache inteiro na RAM, ~15 GB), `--compile`, `--no-pretrained`, `--seed`.

### Execuções acumuladas
Os resultados em `outputs/` são **mantidos entre execuções** (exceto com `--clear`). Cada execução escreve apenas
nos modelos pedidos:

- um modelo treinado de novo tem os arquivos de `outputs/<modelo>/` substituídos; as outras pastas não mudam;
- `summary.json` e `comparison.png` juntam as execuções anteriores com a atual e são atualizados ao fim de cada
  modelo, então uma interrupção preserva o que já terminou (rode de novo só os que faltaram);
- a tabela no terminal lista todos, marcando com `*` os treinados nesta execução.

Cuidados:
- Execuções com opções diferentes (`--ak`, `--subset`, `--seed`, `--epochs`...) aparecem lado a lado, mas podem não
  ser comparáveis. Para separar experimentos, use `--out-dir`.
- `split.json` descreve o split da última execução.
- Se um novo treino for interrompido, o `best.pt` daquele modelo pode já ser o novo enquanto as métricas ainda são
  as antigas; rode esse modelo de novo.
- Para remover um modelo, apague a pasta dele e o `summary.json` (que é reconstruído na próxima execução).

### Limpar os resultados (`--clear`)
`--clear` **apaga todo o conteúdo** da pasta de saída (`outputs/` ou a de `--out-dir`) e então treina. Não pede
confirmação e não tem volta: os checkpoints `best.pt` e subpastas como `outputs/smoke/` também são apagados.

- O cache de imagens (`cache/`) e o dataset não são tocados.
- Nada é apagado se a GPU for inválida ou a pasta do dataset não existir.
- A limpeza é recusada se a pasta contiver o projeto, o diretório atual, o dataset, o cache ou a pasta pessoal, ou
  se não for vazia e não tiver nenhum resultado do projeto (proteção contra um `--out-dir` errado).

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
utils/results.py     junta os resultados de execuções anteriores em summary.json
utils/seed.py        reprodutibilidade
viewer/index.html    visualizador web dos resultados (ver seção abaixo)
```

## Arquiteturas
Todos usam pesos ImageNet (exigem internet na primeira execução) e se escolhem com `--models`.
Sem `--models`, rodam os 7 modelos padrão (marcados com ★); com `--all`, rodam todos os 18 da tabela
(as variantes `_bn` contam separadamente).

Em **ordem ascendente de custo computacional** (GFLOPs por imagem a 224 px, forward; valores aproximados da
documentação do torchvision):

| # | Modelo | GFLOPs | Parâmetros | Obs. |
|---|--------|-------:|-----------:|------|
| 1 | `mobilenet_v3_large` ★ | 0,22 | 5,5 M | |
| 2 | `efficientnet_b0` ★ | 0,39 | 5,3 M | |
| 3 | `resnet18` | 1,8 | 11,7 M | |
| 4 | `efficientnet_b3` ★ | 1,8 | 12,2 M | roda a 300 px (custo real maior) |
| 5 | `densenet121` ★ | 2,8 | 8,0 M | |
| 6 | `resnet34` | 3,7 | 21,8 M | |
| 7 | `resnet50` ★ | 4,1 | 25,6 M | |
| 8 | `convnext_tiny` ★ | 4,5 | 28,6 M | |
| 9 | `swin_t` | 4,5 | 28,3 M | transformer com atenção em janelas |
| 10 | `vgg11` / `vgg11_bn` | 7,6 | 132,9 M | |
| 11 | `vgg13` / `vgg13_bn` | 11,3 | 133,0 M | |
| 12 | `vgg16` / `vgg16_bn` | 15,5 | 138,4 M | |
| 13 | `vit_b_16` ★ | 17,6 | 86,6 M | exige 224 px |
| 14 | `vgg19` / `vgg19_bn` | 19,6 | 143,7 M | |

As variantes `_bn` têm BatchNorm (treinam mais estável) e custo praticamente igual. GFLOPs não é tempo de
treino: VGG tem muitos parâmetros e usa bastante VRAM (batch padrão 32), e DenseNet/EfficientNet costumam ser
mais lentos do que seus GFLOPs sugerem. A ordem é uma referência, não um benchmark medido nesta máquina.

## Visualizador de resultados (`viewer/index.html`)
Página única (sem dependências) para explorar o que o `main.py` gera em `outputs/`: tabela comparativa
ordenável, ranking por métrica, gráfico de barras por modelo (gerado dos `.json`, com métricas liga/desliga na legenda), curvas por época sobrepostas, detalhe por modelo (métricas, matriz de
confusão, imagens `history/roc/confusion.png`, torchinfo), visualização dos `.json` e download de um
**JSON combinado** com métricas e histórico de todos os modelos.

O botão **Salvar snapshot** gera um único `.html` com os dados e imagens carregados e a visão atual (aba, limiar,
métricas ligadas etc.). O arquivo abre em qualquer navegador, sem servidor nem a pasta `outputs/`.

Duas formas de abrir:
```bash
# 1) com servidor (carrega ../outputs/ sozinho); na raiz do projeto:
python -m http.server 8000     # depois abra http://localhost:8000/viewer/

# 2) sem servidor: abra viewer/index.html no navegador e clique em "Abrir pasta outputs"
```
Se o treino roda em outra máquina: `ssh -L 8000:localhost:8000 usuario@servidor`, rode o servidor lá e abra
`http://localhost:8000/viewer/` localmente (ou copie `outputs/` e use a forma 2).

## Data augmentation
Implementada em `data/augment.py` (`augment_train`), roda na GPU, só no **treino**. Cada imagem do batch recebe
seus próprios valores aleatórios, sorteados de novo a cada época. `U(a, b)` = sorteio uniforme entre `a` e `b`.

Ordem de aplicação:

| # | Etapa | Valores | Probabilidade |
|---|-------|---------|--------------:|
| 1 | Conversão `uint8` → `float` | divisão por 255 (faixa 0–1) | sempre |
| 2 | Redimensionamento prévio | 450 → `int(1,15 × tamanho)` (257 px para 224; 345 px para 300), bilinear com antialias | sempre |
| 3a | Flip horizontal | espelhamento | 50% |
| 3b | Rotação | ângulo `U(-180°, 180°)` | 100% |
| 3c | Zoom | fração do campo de visão mantida `U(0,7, 1,0)`, ou seja, zoom de 1,0× a ≈1,43× | 100% |
| 3d | Translação | `U(-0,1, 0,1)` em x e em y, em coordenadas de -1 a 1 (até ±5% do lado da imagem) | 100% |
| 4 | Brilho | `x · f`, com `f = U(0,8, 1,2)` | 100% |
| 5 | Contraste | `(x - m) · f + m`, com `m` = média da imagem (todos os canais) e `f = U(0,8, 1,2)` | 100% |
| 6 | Saturação | `(x - g) · f + g`, com `g` = média dos canais RGB em cada pixel e `f = U(0,8, 1,2)` | 100% |
| 7 | Recorte de valores | `clamp(0, 1)`, apenas aqui (não entre as etapas 4–6) | sempre |
| 8 | Normalização ImageNet | média `(0,485, 0,456, 0,406)`, desvio `(0,229, 0,224, 0,225)` | sempre |

- As etapas **3a–3d** formam **uma única transformação afim** (`affine_grid` + `grid_sample`), então há só uma
  interpolação: bilinear, com preenchimento por **reflexão** nas bordas. A saída já tem o tamanho de entrada do modelo.
- O flip é só horizontal; junto com a rotação livre, ele também cobre o espelhamento vertical.
- O tamanho de entrada é 224 px, exceto `efficientnet_b3` (300 px); `--img-size` altera.
- **Validação e teste** (`preprocess_eval`) não têm augmentation: divisão por 255 → redimensionamento direto para o
  tamanho do modelo (bilinear com antialias) → normalização ImageNet.
- Antes de tudo, o cache já deixou as imagens quadradas em 450 px (ver [Dataset](#dataset)); isso não é augmentation.
- Não são usados: mudança de matiz (hue), blur, ruído, random erasing, MixUp, CutMix nem test-time augmentation.

## Decisões de projeto
- **Saída de 1 logit** com `BCEWithLogitsLoss(pos_weight = n_benigno / n_maligno)` para o desbalanceamento.
- **Augmentation na GPU**: a CPU do alvo é fraca; o DataLoader só entrega tensores uint8 e
  rotação, flip, zoom, brilho, contraste e saturação são aplicados por amostra no batch em `cuda`
  (detalhes em [Data augmentation](#data-augmentation)).
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
