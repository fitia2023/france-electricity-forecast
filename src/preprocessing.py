import re
import unicodedata
import polars as pl
from src.config import PROCESSED_DATA_DIR, RAW_ENEDIS_FILE


def normalize_name(col: str) -> str:
    """Nettoie les noms de colonnes même en présence de caractères mal encodés ()."""
    col_ascii = unicodedata.normalize("NFKD", col).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", col_ascii.lower()).strip("_")


def detect_separator(filepath) -> str:
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline()
    return ";" if header.count(";") > header.count(",") else ","


def find_col(columns: list[str], keywords: list[str]) -> str:
    """Trouve dynamiquement une colonne via des fragments de mots robustes aux accents."""
    for col in columns:
        if all(kw in col for kw in keywords):
            return col
    raise KeyError(f"Colonne introuvable avec {keywords} parmi {columns}")


def run_pipeline() -> None:
    if not RAW_ENEDIS_FILE.exists():
        raise FileNotFoundError(f"Fichier brut introuvable : {RAW_ENEDIS_FILE}")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    sep = detect_separator(RAW_ENEDIS_FILE)

    # Lecture lazy avec gestion tolérante de l'encodage
    lf = pl.scan_csv(
        RAW_ENEDIS_FILE,
        separator=sep,
        infer_schema_length=0,
        encoding="utf8-lossy",
    )

    raw_cols = lf.collect_schema().names()
    rename_map = {c: normalize_name(c) for c in raw_cols}
    lf = lf.rename(rename_map)
    cols = list(rename_map.values())

    # Mapping robuste sur tes 19 colonnes ENEDIS
    col_annee = find_col(cols, ["ann"])
    col_iris = find_col(cols, ["code", "iris"])
    col_commune = find_col(cols, ["code", "commune"])
    col_segment = find_col(cols, ["segment"])
    col_logements = find_col(cols, ["nombre", "logements"])
    col_conso = find_col(cols, ["consommation", "totale"])
    col_epci = find_col(cols, ["epci"])
    col_dept = find_col(cols, ["code", "partement"])
    col_region = find_col(cols, ["code", "gion"])

    clean_lf = (
        lf.select(
            [
                pl.col(col_annee).cast(pl.Int32, strict=False).alias("annee"),
                pl.col(col_region).cast(pl.String).str.strip_chars().str.zfill(2).alias("code_region"),
                pl.col(col_dept).cast(pl.String).str.strip_chars().alias("code_departement"),
                pl.col(col_epci).cast(pl.String).str.strip_chars().alias("code_epci"),
                pl.col(col_commune).cast(pl.String).str.strip_chars().alias("code_commune"),
                pl.col(col_iris).cast(pl.String).str.strip_chars().alias("code_iris"),
                pl.col(col_segment).cast(pl.String).str.strip_chars().alias("segment"),
                pl.col(col_logements).str.replace(",", ".").cast(pl.Float64, strict=False).alias("nb_logements"),
                pl.col(col_conso).str.replace(",", ".").cast(pl.Float64, strict=False).alias("conso_totale_mwh"),
            ]
        )
        .filter(
            pl.col("segment").str.to_uppercase().str.contains("RESIDENTIEL")
            & pl.col("annee").is_not_null()
            & pl.col("conso_totale_mwh").is_not_null()
            & (pl.col("conso_totale_mwh") > 0)
            & (pl.col("nb_logements") > 0)
        )
    )

    print("Exécution du pipeline d'agrégation en cours...")

    # 1. Table macro annuelle (avec compteurs de couverture géographique)
    df_france = (
        clean_lf.group_by("annee")
        .agg(
            pl.len().alias("nb_adresses"),
            pl.n_unique("code_region").alias("nb_regions"),
            pl.n_unique("code_departement").alias("nb_departements"),
            pl.n_unique("code_iris").alias("nb_iris"),
            pl.col("nb_logements").sum().alias("nb_logements_total"),
            pl.col("conso_totale_mwh").sum().alias("conso_totale_mwh"),
        )
        .with_columns(
            (pl.col("conso_totale_mwh") / 1_000_000).alias("conso_totale_twh"),
            (pl.col("conso_totale_mwh") / pl.col("nb_logements_total")).alias("conso_moyenne_logement_mwh"),
        )
        .sort("annee")
        .collect()
    )
    df_france.write_parquet(PROCESSED_DATA_DIR / "france_annuel.parquet")
    print(f"[OK] france_annuel.parquet généré ({df_france.height} lignes)")

    # 2. Table par Région
    df_regions = (
        clean_lf.group_by(["annee", "code_region"])
        .agg(
            pl.len().alias("nb_adresses"),
            pl.n_unique("code_departement").alias("nb_departements"),
            pl.col("nb_logements").sum().alias("nb_logements_total"),
            pl.col("conso_totale_mwh").sum().alias("conso_totale_mwh"),
        )
        .with_columns(
            (pl.col("conso_totale_mwh") / 1_000_000).alias("conso_totale_twh"),
            (pl.col("conso_totale_mwh") / pl.col("nb_logements_total")).alias("conso_moyenne_logement_mwh"),
        )
        .sort(["code_region", "annee"])
        .collect()
    )
    df_regions.write_parquet(PROCESSED_DATA_DIR / "regions_annuel.parquet")
    print(f"[OK] regions_annuel.parquet généré ({df_regions.height} lignes)")

    # 3. Table par Département
    df_depts = (
        clean_lf.group_by(["annee", "code_region", "code_departement"])
        .agg(
            pl.len().alias("nb_adresses"),
            pl.col("nb_logements").sum().alias("nb_logements_total"),
            pl.col("conso_totale_mwh").sum().alias("conso_totale_mwh"),
        )
        .with_columns(
            (pl.col("conso_totale_mwh") / pl.col("nb_logements_total")).alias("conso_moyenne_logement_mwh"),
        )
        .sort(["code_departement", "annee"])
        .collect()
    )
    df_depts.write_parquet(PROCESSED_DATA_DIR / "departements_annuel.parquet")
    print(f"[OK] departements_annuel.parquet généré ({df_depts.height} lignes)")

    # 4. Table par IRIS (pour approche panel / Machine Learning)
    df_iris = (
        clean_lf.group_by(["annee", "code_region", "code_departement", "code_commune", "code_iris"])
        .agg(
            pl.len().alias("nb_adresses"),
            pl.col("nb_logements").sum().alias("nb_logements_total"),
            pl.col("conso_totale_mwh").sum().alias("conso_totale_mwh"),
        )
        .with_columns(
            (pl.col("conso_totale_mwh") / pl.col("nb_logements_total")).alias("conso_moyenne_logement_mwh"),
        )
        .sort(["code_iris", "annee"])
        .collect()
    )
    df_iris.write_parquet(PROCESSED_DATA_DIR / "iris_annuel.parquet")
    print(f"[OK] iris_annuel.parquet généré ({df_iris.height} lignes)")


if __name__ == "__main__":
    run_pipeline()
