import re
import unicodedata
import polars as pl
from src.config import PROCESSED_DATA_DIR, RAW_ECO2_FILE


def normalize_name(col: str) -> str:
    col_ascii = unicodedata.normalize("NFKD", col).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", col_ascii.lower()).strip("_")


def run_pipeline() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    lf = pl.scan_csv(
        RAW_ECO2_FILE,
        separator=";",
        infer_schema_length=0,
        encoding="utf8-lossy",
    )

    raw_cols = lf.collect_schema().names()
    rename_map = {c: normalize_name(c) for c in raw_cols}
    lf = lf.rename(rename_map)
    cols = list(rename_map.values())

    col_date = next(c for c in cols if c == "date")
    col_heure = next(c for c in cols if c == "heure")
    col_conso = next(c for c in cols if "consommation" in c and "mw" in c)
    opt_perimetre = next((c for c in cols if "perimetre" in c), None)

    if opt_perimetre:
        lf = lf.filter(pl.col(opt_perimetre).str.to_lowercase().str.contains("france"))

    # Formatage strict : parsing Date + Heure, suppression des pas 15-min vides
    clean_lf = (
        lf.select(
            [
                pl.col(col_date).str.to_date("%Y-%m-%d", strict=False).alias("date"),
                pl.col(col_heure).alias("heure"),
                pl.col(col_conso).cast(pl.Float64, strict=False).alias("conso_mw"),
            ]
        )
        .filter(pl.col("date").is_not_null() & pl.col("conso_mw").is_not_null() & (pl.col("conso_mw") > 0))
        .with_columns(
            (pl.col("date").cast(pl.String) + " " + pl.col("heure")).str.to_datetime("%Y-%m-%d %H:%M", strict=False).alias("datetime")
        )
        .sort("datetime")
    )

    # 1. Série Journalière (GWh = puissance moyenne MW * 24h / 1000)
    df_jour = (
        clean_lf.group_by("date")
        .agg(
            pl.col("conso_mw").mean().alias("puissance_moy_mw"),
            pl.col("conso_mw").max().alias("pic_puissance_mw"),
            pl.len().alias("nb_points_mesure"),
        )
        .with_columns((pl.col("puissance_moy_mw") * 24.0 / 1000.0).alias("conso_gwh"))
        .sort("date")
        .collect()
    )
    df_jour.write_parquet(PROCESSED_DATA_DIR / "eco2_france_jour.parquet")
    print(f"[OK] eco2_france_jour.parquet : {df_jour.height} jours ({df_jour['date'].min()} -> {df_jour['date'].max()})")

    # 2. Série Hebdomadaire (Lundi -> Dimanche, GWh = puissance moyenne MW * 168h / 1000)
    # On ne garde que les semaines complètes (7 jours) pour éviter les effets de bord
    df_hebdo = (
        df_jour.lazy()
        .with_columns(pl.col("date").dt.truncate("1w").alias("semaine"))
        .group_by("semaine")
        .agg(
            pl.col("conso_gwh").sum().alias("conso_gwh"),
            pl.col("puissance_moy_mw").mean().alias("puissance_moy_mw"),
            pl.len().alias("nb_jours"),
        )
        .filter(pl.col("nb_jours") == 7)
        .sort("semaine")
        .collect()
    )
    df_hebdo.write_parquet(PROCESSED_DATA_DIR / "eco2_france_hebdo.parquet")
    print(f"[OK] eco2_france_hebdo.parquet : {df_hebdo.height} semaines complètes")


if __name__ == "__main__":
    run_pipeline()
