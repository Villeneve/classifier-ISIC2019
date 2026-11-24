"""Criação de modelos torchvision com cabeça de 1 logit (maligno)."""
import torch.nn as nn
import torchvision.models as tvm


def _last_linear(model):
    """Retorna (módulo_pai, nome_do_atributo, Linear) do último nn.Linear."""
    found = None
    for name, mod in model.named_modules():
        for child_name, child in mod.named_children():
            if isinstance(child, nn.Linear):
                found = (mod, child_name, child)
    return found


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
