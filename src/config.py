from pathlib import Path

# Racine du projet détectée automatiquement quel que soit l'OS
ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FIGURES_DIR = ROOT_DIR / "report" / "figures"

# Fichier brut attendu
RAW_ENEDIS_FILE = RAW_DATA_DIR / "consommation-annuelle-residentielle-par-adresse.csv"
