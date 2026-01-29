import pandas as pd
import numpy as np
import os
import re

def clean_price(val):
    if pd.isna(val): return np.nan
    # Nettoyage basique : on garde chiffres et point
    clean = str(val).replace("€", "").replace(",", ".").replace(" ", "")
    try:
        match = re.search(r"(\d+\.?\d*)", clean)
        if match: return float(match.group(1))
    except: pass
    return np.nan

def clean_rating(val):
    if pd.isna(val): return np.nan
    try:
        clean = str(val).replace(",", ".")
        match = re.search(r"(\d+\.?\d*)", clean)
        if match: return float(match.group(1))
    except: pass
    return np.nan

def clean_reviews(val):
    if pd.isna(val): return 0
    # On ne garde que les chiffres
    digits = re.sub(r"\D", "", str(val))
    if digits: return int(digits)
    return 0

def main():
    # Chargement des données
    input_file = "data/raw/amazon_raw_data.csv"
    if not os.path.exists(input_file):
        input_file = "data/raw/amazon_temp.csv"
        
    if not os.path.exists(input_file):
        print("Erreur : Aucun fichier de données trouvé.")
        return

    df = pd.read_csv(input_file)
    print(f"Chargement de {len(df)} lignes.")

    # Suppression doublons
    df.drop_duplicates(subset=['url'], inplace=True)

    # Application du nettoyage
    df['price'] = df['price'].apply(clean_price)
    df['rating'] = df['rating'].apply(clean_rating)
    df['review_count'] = df['review_count'].apply(clean_reviews)

    # Gestion des avis manquants (Important pour la suite)
    if 'first_review' in df.columns:
        df['first_review'] = df['first_review'].fillna("")
    else:
        df['first_review'] = ""

    # Suppression des produits inexploitables (sans prix)
    df.dropna(subset=['price'], inplace=True)

    # Imputation simple pour les notes manquantes
    df['rating'] = df['rating'].fillna(df['rating'].mean())

    # Sauvegarde
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/amazon_clean.csv", index=False)
    print(f"Nettoyage terminé. {len(df)} produits valides sauvegardés.")

if __name__ == "__main__":
    main()