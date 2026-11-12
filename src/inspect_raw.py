import polars as pl
from src.config import RAW_ENEDIS_FILE


def detect_separator(filepath) -> str:
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline()
    return ";" if header.count(";") > header.count(",") else ","


def inspect():
    sep = detect_separator(RAW_ENEDIS_FILE)
    print(f"Séparateur détecté : '{sep}'")

    # Lecture des 5 premières lignes uniquement
    sample = pl.read_csv(RAW_ENEDIS_FILE, separator=sep, n_rows=5, infer_schema_length=0)
    print("\n--- COLONNES EXACTES ---")
    for i, col in enumerate(sample.columns):
        print(f"{i:02d}: {col}")

    print("\n--- APERÇU DES 3 PREMIÈRES LIGNES ---")
    print(sample.head(3))

    # Scan global ultra-rapide (lazy) pour compter les lignes et voir les années disponibles
    q = pl.scan_csv(RAW_ENEDIS_FILE, separator=sep, infer_schema_length=0)
    total_rows = q.select(pl.len()).collect().item()
    print(f"\nNombre total de lignes : {total_rows:,}")

    year_col = [c for c in sample.columns if "ann" in c.lower() or "year" in c.lower()][0]
    years = (
        q.select(pl.col(year_col).unique().sort())
        .collect()
        .to_series()
        .to_list()
    )
    print(f"Années couvertes ({year_col}) : {years}")


if __name__ == "__main__":
    inspect()
