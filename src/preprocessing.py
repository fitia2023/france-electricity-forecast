from src.config import PROCESSED_DATA_DIR, RAW_ENEDIS_FILE


def run_pipeline() -> None:
    """Pipeline d'ingestion et d'agrégation des données ENEDIS."""
    if not RAW_ENEDIS_FILE.exists():
        raise FileNotFoundError(f"Fichier brut introuvable : {RAW_ENEDIS_FILE}")
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Prêt pour le traitement des données.")


if __name__ == "__main__":
    run_pipeline()
