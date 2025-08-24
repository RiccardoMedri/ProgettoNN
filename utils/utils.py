import os
import torch
import json

def load_config(config_path: str) -> dict:
    """
    Carica la configurazione JSON da file.
    Expected keys in config.json:
      - mean: [float, float, float]
      - std: [float, float, float]
      - image_size: [width, height]
      - augment: dict of augmentation params (optional)
    """
    with open(config_path, 'r') as f:
        return json.load(f)
    
def save_checkpoint(state: dict, filepath: str):
  """
  Salva su disco lo stato del training.
  - state: dict contenente almeno
      - 'epoch'
      - 'model_state_dict'
      - 'optimizer_state_dict'
      - eventuali altri campi (es. 'best_map', 'config', ecc.)
  - filepath: es. "outputs/weights/ckpt_epoch10.pth"
  """
  # Assicuriamoci che la cartella esista
  os.makedirs(os.path.dirname(filepath), exist_ok=True)
  torch.save(state, filepath)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    filepath: str
) -> dict:
    """
    Carica modello e ottimizzatore da checkpoint.
    Ritorna il dict salvato, contenente ad es. 'epoch' e 'best_map'.
    """
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    # carica pesi modello (ignora gli eventuali mismatch di shape)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    # carica stato ottimizzatore
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint