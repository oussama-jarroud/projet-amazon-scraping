import pandas as pd
import numpy as np
import re
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import os

os.makedirs("data/final", exist_ok=True)

def load_data():
    try:
        df = pd.read_csv("data/processed/amazon_clean.csv")
        print(f"Chargement : {len(df)} produits.")
        return df
    except FileNotFoundError:
        print("Erreur : Fichier introuvable.")
        return None

def add_nlp_features(df):
    """
    Analyse de sentiment sur les avis clients et stats sur le titre.
    """
    print("--- 1. Extraction Features NLP ---")
    
    # Gestion des valeurs manquantes dans les avis
    if 'first_review' not in df.columns:
        df['first_review'] = ""
    df['first_review'] = df['first_review'].fillna("")

    # Analyse de sentiment sur le premier commentaire (Review)
    # C'est ici qu'on utilise TextBlob sur l'avis et non le titre
    df['review_sentiment'] = df['first_review'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df['review_subjectivity'] = df['first_review'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
    
    # Statistiques basiques sur le titre
    df['title_length'] = df['title'].apply(lambda x: len(str(x)))
    df['word_count'] = df['title'].apply(lambda x: len(str(x).split()))
    
    return df

def add_tfidf_features(df, max_features=15):
    """
    TF-IDF sur les titres pour capter les mots clés importants.
    """
    print("--- 2. TF-IDF sur les titres ---")
    
    # On nettoie un peu le titre avant
    clean_titles = df['title'].astype(str).str.lower().str.replace(r'[^\w\s]', '', regex=True)
    
    tfidf = TfidfVectorizer(stop_words='english', max_features=max_features)
    
    try:
        tfidf_matrix = tfidf.fit_transform(clean_titles)
        feature_names = tfidf.get_feature_names_out()
        
        # Ajout des colonnes au DataFrame
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tfidf_{w}" for w in feature_names])
        df = pd.concat([df, tfidf_df], axis=1)
        
    except ValueError:
        print("Pas assez de données pour TF-IDF.")
    
    return df

def add_brand_feature(df):
    """
    Extraction de la marque (1er mot du titre).
    """
    print("--- 3. Gestion des marques ---")
    
    df['brand'] = df['title'].apply(lambda x: str(x).split()[0].upper().strip())
    
    # On garde les 7 marques les plus fréquentes
    top_brands = df['brand'].value_counts().nlargest(7).index
    df['brand_encoded'] = df['brand'].apply(lambda x: x if x in top_brands else 'OTHER')
    
    # One-Hot Encoding
    df = pd.get_dummies(df, columns=['brand_encoded'], prefix='brand')
    df.drop(columns=['brand'], inplace=True)
    
    return df

def add_domain_features(df):
    """
    Variables spécifiques au produit (Regex sur le titre).
    """
    print("--- 4. Features Techniques ---")
    
    title_lower = df['title'].str.lower()
    
    # Détection de mots clés techniques dans le titre
    df['is_mechanical'] = title_lower.str.contains('mecanique|mechanical|switch').astype(int)
    df['is_wireless'] = title_lower.str.contains('sans fil|wireless|bluetooth').astype(int)
    df['is_rgb'] = title_lower.str.contains('rgb|led|lumiere').astype(int)
    df['is_azerty'] = title_lower.str.contains('azerty|fr').astype(int)
    
    # Log du prix pour réduire l'impact des prix extrêmes
    df['price_log'] = np.log1p(df['price'])
    
    return df

def define_target(df):
    """
    Création de la variable cible 'is_successful'.
    """
    print("--- 5. Définition de la Target ---")
    
    review_median = df['review_count'].median()
    
    # Un produit est un "Succès" s'il a beaucoup d'avis (> médiane) et une bonne note (>= 4)
    def is_hit(row):
        if row['review_count'] > review_median and row['rating'] >= 4.0:
            return 1
        return 0
            
    df['is_successful'] = df.apply(is_hit, axis=1)
    return df

def main():
    df = load_data()
    if df is None: return

    # Pipeline de création de features
    df = add_nlp_features(df)
    df = add_brand_feature(df)
    df = add_tfidf_features(df)
    df = add_domain_features(df)
    df = define_target(df)

    # Suppression des colonnes textes brutes avant export pour le ML
    cols_to_drop = ['url', 'title', 'first_review']
    df_final = df.drop(columns=cols_to_drop, errors='ignore')

    output_path = "data/final/amazon_ml_ready.csv"
    df_final.to_csv(output_path, index=False)
    print(f"Terminé. Fichier prêt : {output_path}")

if __name__ == "__main__":
    main()