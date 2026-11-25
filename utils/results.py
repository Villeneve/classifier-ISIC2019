"""Resultados acumulados em outputs/: uma execução só substitui os modelos que ela treinou."""
import json
import os


def _read_json(path):
    """Conteúdo do .json, ou None se não existir ou estiver inválido."""
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        print(f"aviso: {path} ignorado ({e})")
        return None


def merge_summary(out_dir, new):
    """Atualiza outputs/summary.json e retorna o resumo completo {modelo: métricas}.

    Junta, em ordem de prioridade crescente: o summary.json existente, os <modelo>/metrics.json presentes
    na pasta e os resultados desta execução (`new`).
    """
    path = os.path.join(out_dir, "summary.json")
    merged = _read_json(path) or {}
    for name in sorted(os.listdir(out_dir)):
        folder = os.path.join(out_dir, name)
        if not os.path.isdir(folder):
            continue
        res = _read_json(os.path.join(folder, "metrics.json"))
        if res is not None:
            merged[name] = res
    merged.update(new)

    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(merged, f, indent=2)
    os.replace(tmp, path)  # troca atômica: uma interrupção não corrompe o resumo
    return merged
