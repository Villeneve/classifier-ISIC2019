"""Criação de modelos torchvision com cabeça de 1 logit (maligno)."""
import torch.nn as nn
import torchvision.models as tvm


def _last_linear(model):
    """Retorna (módulo_pai, nome_do_atributo, Linear) do último nn.Linear (a cabeça)."""
    # named_modules percorre na ordem de registro, então o último Linear da lista é a cabeça.
    name, linear = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)][-1]
    parent, _, attr = name.rpartition(".")
    return model.get_submodule(parent), attr, linear


def build_model(name, pretrained=True):
    weights = tvm.get_model_weights(name).DEFAULT if pretrained else None
    model = tvm.get_model(name, weights=weights)
    parent, attr, lin = _last_linear(model)
    setattr(parent, attr, nn.Linear(lin.in_features, 1))
    return model


def set_backbone_trainable(model, trainable):
    """Congela/descongela tudo exceto a cabeça (último Linear)."""
    _, _, head = _last_linear(model)
    head_ids = {id(p) for p in head.parameters()}
    for p in model.parameters():
        p.requires_grad = trainable or id(p) in head_ids
