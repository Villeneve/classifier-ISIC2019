"""Resultados acumulados em outputs/: uma execução só substitui os modelos que ela treinou."""
import json
import os
import shutil


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


def clear_results(out_dir, model_names, protected=()):
    """Apaga todo o conteúdo de `out_dir` (a pasta em si fica). Retorna quantos itens foram removidos.

    Como a operação não tem volta, recusa (ValueError) uma pasta que contenha algum caminho `protected`
    ou a pasta pessoal, e uma pasta não vazia que não pareça conter resultados deste projeto.
    """
    if not os.path.isdir(out_dir):
        return 0
    root = os.path.realpath(out_dir)
    for p in (*protected, os.path.expanduser("~")):
        if os.path.commonpath([root, os.path.realpath(p)]) == root:
            raise ValueError(f"--clear recusado: {out_dir!r} contém {p!r}")
    entries = os.listdir(root)
    known = {"summary.json", "split.json", "comparison.png", *model_names}
    if entries and not known.intersection(entries):
        raise ValueError(f"--clear recusado: {out_dir!r} não parece uma pasta de resultados")
    for name in entries:
        path = os.path.join(root, name)
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        else:
            os.remove(path)  # arquivo ou link (o destino do link não é tocado)
    return len(entries)


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
