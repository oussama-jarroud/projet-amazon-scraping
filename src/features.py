import pandas as pd
import numpy as np
import re
import os
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer

# --- CONFIGURATION NLP ---
# On tente d'importer l'analyseur français.
# Comme tu as choisi la Méthode 1, ce bloc va fonctionner.
try:
    from textblob_fr import PatternAnalyzer
    HAS_TEXTBLOB_FR = True
    print("Succès : Librairie 'textblob-fr' détectée.")
except ImportError:
    HAS_TEXTBLOB_FR = False
    print("Attention : 'textblob-fr' non installé. Le code utilisera un mode dégradé.")

os.makedirs("data/final", exist_ok=True)

def load_data():
    """Charge les données nettoyées."""
    try:
        df = pd.read_csv("data/processed/amazon_clean.csv")
        print(f"Chargement : {len(df)} produits.")
        return df
    except FileNotFoundError:
        print("Erreur : Fichier introuvable. Lancez 'cleaner.py' d'abord.")
        return None

def get_sentiment_score(text):
    """
    Calcule le sentiment (positif/négatif) d'un texte français.
    Utilise 'textblob-fr' (Méthode 1).
    """
    text = str(text)
    if not text or text.lower() == "nan":
        return 0.0

    if HAS_TEXTBLOB_FR:
        # C'est la ligne clé pour la Méthode 1
        blob = TextBlob(text, analyzer=PatternAnalyzer())
        return blob.sentiment[0]  # Retourne la polarité (-1 à 1)
    
    else:
        # Fallback de sécurité (au cas où l'installation a échoué)
        return TextBlob(text).sentiment.polarity

def add_nlp_features(df):
    """
    Feature Engineering sur le texte (NLP).
    """
    print("--- 1. Extraction Features NLP (Sentiment & Stats) ---")
    
    # Gestion des valeurs nulles
    if 'first_review' not in df.columns:
        df['first_review'] = ""
    df['first_review'] = df['first_review'].fillna("")

    # Analyse de sentiment (Utilise la fonction configurée plus haut)
    df['review_sentiment'] = df['first_review'].apply(get_sentiment_score)
    
    # Subjectivité (Indique si c'est une opinion ou un fait)
    # TextBlob standard gère bien la subjectivité même sur des mots latins
    df['review_subjectivity'] = df['first_review'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
    
    # Statistiques sur la longueur du titre
    df['title_length'] = df['title'].apply(lambda x: len(str(x)))
    df['word_count'] = df['title'].apply(lambda x: len(str(x).split()))
    
    return df

def add_tfidf_features(df, max_features=15):
    """
    TF-IDF : Trouve les mots clés importants dans les titres.
    """
    print("--- 2. TF-IDF sur les titres ---")
    
    # Nettoyage rapide du texte
    clean_titles = df['title'].astype(str).str.lower().str.replace(r'[^\w\s]', '', regex=True)
    
    try:
        # Utilisation des stop_words français
        tfidf = TfidfVectorizer(stop_words='french', max_features=max_features)
        tfidf_matrix = tfidf.fit_transform(clean_titles)
        
        feature_names = tfidf.get_feature_names_out()
        
        # Création du DataFrame avec les mots clés
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tfidf_{w}" for w in feature_names])
        
        # Reset des index pour coller parfaitement
        df = df.reset_index(drop=True)
        tfidf_df = tfidf_df.reset_index(drop=True)
        
        df = pd.concat([df, tfidf_df], axis=1)
        
    except ValueError:
        print("Pas assez de données pour générer le TF-IDF.")
    except Exception as e:
        print(f"Erreur TF-IDF : {e}. On continue sans.")
    
    return df

def add_brand_feature(df):
    """
    Extrait la marque et fait un One-Hot Encoding.
    """
    print("--- 3. Gestion des marques ---")
    
    # On prend le premier mot du titre comme marque
    df['brand'] = df['title'].apply(lambda x: str(x).split()[0].upper().strip())
    
    # On garde les 10 plus fréquentes, les autres deviennent "OTHER"
    top_brands = df['brand'].value_counts().nlargest(10).index
    df['brand_encoded'] = df['brand'].apply(lambda x: x if x in top_brands else 'OTHER')
    
    # Transformation en colonnes binaires (0 ou 1)
    df = pd.get_dummies(df, columns=['brand_encoded'], prefix='brand')
    df.drop(columns=['brand'], inplace=True)
    
    return df

def add_domain_features(df):
    """
    Crée des variables basées sur des mots clés techniques (Regex).
    """
    print("--- 4. Features Techniques ---")
    
    title_lower = df['title'].str.lower()
    
    # Détection de mots clés
    df['is_mechanical'] = title_lower.str.contains('mecanique|mécanique|mechanical|switch').astype(int)
    df['is_wireless'] = title_lower.str.contains('sans fil|wireless|bluetooth|rechargeable').astype(int)
    df['is_rgb'] = title_lower.str.contains('rgb|led|lumiere|rétroéclairé').astype(int)
    df['is_azerty'] = title_lower.str.contains('azerty|fr|français').astype(int)
    
    # Log du prix (Meilleur pour les modèles ML)
    df['price_log'] = np.log1p(df['price'])
    
    return df

def define_target(df):
    """
    Définit ce qu'est un produit 'Succès' (Variable Y).
    """
    print("--- 5. Définition de la Target ---")
    
    if 'review_count' not in df.columns or 'rating' not in df.columns:
        print("Erreur critique: Colonnes manquantes.")
        return df

    # Critères dynamiques
    review_median = df['review_count'].median()
    rating_threshold = 4.0
    
    print(f"Seuil de succès : > {review_median} avis ET note >= {rating_threshold}/5")
    
    def is_hit(row):
        if row['review_count'] > review_median and row['rating'] >= rating_threshold:
            return 1
        return 0
            
    df['is_successful'] = df.apply(is_hit, axis=1)
    
    print(f"Distribution : {df['is_successful'].value_counts(normalize=True)}")
    return df

def main():
    df = load_data()
    if df is None: return

    # Exécution du pipeline
    df = add_nlp_features(df)
    df = add_brand_feature(df)
    df = add_tfidf_features(df)
    df = add_domain_features(df)
    df = define_target(df)

    # Nettoyage avant export ML
    # On retire les textes bruts qui ne servent plus au modèle mathématique
    cols_to_drop = ['url', 'title', 'first_review']
    df_final = df.drop(columns=cols_to_drop, errors='ignore')

    output_path = "data/final/amazon_ml_ready.csv"
    df_final.to_csv(output_path, index=False)
    print(f"\n[SUCCÈS] Fichier prêt pour le ML : {output_path}")

if __name__ == "__main__":
    main()