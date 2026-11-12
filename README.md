# Prévision de la consommation annuelle d'électricité résidentielle en France (ENEDIS)

Projet de Master — Séries Temporelles.

## 1. Installation
1. Cloner le dépôt et installer les dépendances (`poetry install` ou `pip install -r requirements.txt`).
2. Télécharger le fichier CSV brut ENEDIS depuis le lien partagé du groupe et le placer dans `data/raw/` sous le nom exact :
   `consommation-annuelle-residentielle-par-adresse.csv`

## 2. Générer les données propres (`.parquet`)
```bash
python -m src.preprocessing
