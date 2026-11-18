from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FIGURES_DIR = ROOT_DIR / "report" / "figures"

# Détecte automatiquement le fichier CSV présent dans data/raw/
csv_files = list(RAW_DATA_DIR.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"Aucun fichier .csv trouvé dans {RAW_DATA_DIR}")
RAW_ECO2_FILE = csv_files[0]
