import os
import yaml

def _load_names():
    """
    Tenta di caricare la lista di nomi classi da un file YAML standard
    (es. config/dataset.yaml contenente la chiave 'names').
    """
    base = os.path.dirname(os.path.dirname(__file__))
    cfg_path = os.path.join(base, "config", "dataset.yaml")

    with open(cfg_path, "r") as f:
        ds_cfg = yaml.safe_load(f)

    names = ds_cfg.get("names")
    if names is None:
        raise KeyError(
            f"File {cfg_path} non contiene la chiave 'names'. "
            "Aggiungi ad es.: names: ['classe0', 'classe1', …]"
        )
    return names


#: Lista delle etichette, es. ['person', 'car', 'bike', …]
class_names = _load_names()
