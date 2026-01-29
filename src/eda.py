import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from wordcloud import WordCloud

# Configuration graphique
sns.set_theme(style="whitegrid", context="talk")
os.makedirs("reports/figures", exist_ok=True)

def load_data():
    try:
        df = pd.read_csv("data/processed/amazon_clean.csv")
        print(f"[INFO] Dataset chargé : {df.shape[0]} lignes.")
        return df
    except FileNotFoundError:
        print("[ERREUR] Fichier introuvable. Lancez cleaner.py d'abord.")
        return None

def plot_segmentation(df):
    """
    Segmentation du marché par gammes de prix (Bas, Moyen, Haut).
    Objectif : Voir si payer plus cher garantit une meilleure note.
    """
    # Création de segments de prix basés sur les quantiles
    df['segment'] = pd.qcut(df['price'], q=3, labels=["Budget", "Milieu de Gamme", "Premium"])
    
    plt.figure(figsize=(12, 7))
    sns.boxplot(x='segment', y='rating', data=df, palette="Blues", showfliers=False)
    sns.stripplot(x='segment', y='rating', data=df, color=".3", alpha=0.5)
    
    plt.title('Distribution des notes par gamme de prix')
    plt.xlabel('Gamme de Prix')
    plt.ylabel('Note Client')
    plt.tight_layout()
    plt.savefig("reports/figures/1_segmentation_prix_qualite.png")
    plt.close()
    print("[GRAPH] Segmentation Prix/Qualité générée.")

def plot_market_map(df):
    """
    Cartographie du marché : Prix vs Popularité.
    Permet d'identifier les produits 'Outliers' (très populaires).
    """
    plt.figure(figsize=(12, 8))
    
    # Échelle log pour mieux visualiser les écarts de popularité
    sns.scatterplot(
        data=df, 
        x='price', 
        y='review_count', 
        hue='rating', 
        size='review_count',
        sizes=(50, 400),
        palette='viridis',
        alpha=0.8,
        edgecolor='black'
    )
    
    plt.yscale('log')
    plt.title('Cartographie du Marché (Prix vs Popularité)')
    plt.xlabel('Prix (€)')
    plt.ylabel('Nombre d\'avis (Échelle Log)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig("reports/figures/2_cartographie_marche.png")
    plt.close()
    print("[GRAPH] Cartographie générée.")

def plot_wordcloud(df):
    """
    Nuage de mots basé sur les titres des produits.
    """
    # Concaténation de tous les titres
    text = " ".join(str(title) for title in df['title'])
    
    wordcloud = WordCloud(
        width=1600, 
        height=800, 
        background_color='white',
        colormap='magma',
        max_words=100
    ).generate(text)
    
    plt.figure(figsize=(15, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Mots-clés fréquents dans les titres')
    plt.tight_layout()
    plt.savefig("reports/figures/3_wordcloud.png")
    plt.close()
    print("[GRAPH] Nuage de mots généré.")

def plot_pairplot(df):
    """
    Vue d'ensemble des corrélations entre variables numériques.
    """
    cols = ['price', 'rating', 'review_count']
    
    # Création d'une variable binaire pour la couleur
    df['is_top_rated'] = df['rating'] > 4.2
    
    sns.pairplot(df[cols + ['is_top_rated']], hue='is_top_rated', palette='husl', diag_kind='kde')
    plt.savefig("reports/figures/4_pairplot.png")
    plt.close()
    print("[GRAPH] Pairplot généré.")

def main():
    df = load_data()
    if df is not None:
        print("--- Génération des graphiques ---")
        plot_segmentation(df)
        plot_market_map(df)
        plot_wordcloud(df)
        plot_pairplot(df)
        print("\n[FIN] Analyse exploratoire terminée.")

if __name__ == "__main__":
    main()