import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Dossier de sortie
os.makedirs("reports/model_results", exist_ok=True)

def load_data():
    try:
        return pd.read_csv("data/final/amazon_ml_ready.csv")
    except FileNotFoundError:
        print("Erreur : Données introuvables.")
        return None

def prepare_features(df):
    """
    Séparation des features (X) et de la cible (y).
    On retire les variables qu'on ne peut pas connaître avant la mise en vente (notes, avis).
    """
    # Colonnes à exclure : Cible + Données futures + Textes bruts
    drop_cols = ['is_successful', 'rating', 'review_count', 'url', 'title', 'first_review']
    
    # On ne garde que les colonnes qui existent vraiment dans le DF
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    
    X = df.drop(columns=existing_drop_cols)
    y = df['is_successful']
    
    print(f"Features utilisées : {len(X.columns)} variables.")
    return X, y

def train_and_compare(X, y):
    print("--- Entraînement et Comparaison ---")
    
    # Split 80% train / 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. Modèle Baseline : Arbre de décision simple
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    acc_dt = dt.score(X_test, y_test)
    print(f"-> Modèle 1 (Arbre Simple) Accuracy : {acc_dt:.2%}")
    
    # 2. Modèle Avancé : Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    acc_rf = rf.score(X_test, y_test)
    print(f"-> Modèle 2 (Random Forest) Accuracy : {acc_rf:.2%}")
    
    # On retourne le meilleur modèle (Random Forest)
    return rf, X_test, y_test, X_train.columns

def evaluate_model(model, X_test, y_test):
    print("\n--- Évaluation détaillée (Meilleur Modèle) ---")
    
    predictions = model.predict(X_test)
    
    # Rapport de classification
    report = classification_report(y_test, predictions)
    print(report)
    
    # Sauvegarde
    with open("reports/model_results/performance.txt", "w") as f:
        f.write(report)
        
    # Matrice de confusion
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matrice de Confusion')
    plt.ylabel('Réalité')
    plt.xlabel('Prédiction')
    plt.tight_layout()
    plt.savefig("reports/model_results/confusion_matrix.png")
    plt.close()

def plot_feature_importance(model, feature_names):
    """
    Visualisation des variables les plus importantes.
    """
    importance = model.feature_importances_
    df_imp = pd.DataFrame({'feature': feature_names, 'importance': importance})
    df_imp = df_imp.sort_values('importance', ascending=False).head(10)
    
    print("\nTop 5 des facteurs de succès :")
    print(df_imp.head(5))
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=df_imp, palette='viridis')
    plt.title('Importance des variables (Random Forest)')
    plt.xlabel('Poids')
    plt.tight_layout()
    plt.savefig("reports/model_results/feature_importance.png")
    plt.close()

def main():
    # 1. Chargement
    df = load_data()
    if df is None: return
    
    # 2. Préparation
    X, y = prepare_features(df)
    
    # 3. Entraînement et choix du modèle
    model, X_test, y_test, feature_names = train_and_compare(X, y)
    
    # 4. Évaluation finale
    evaluate_model(model, X_test, y_test)
    plot_feature_importance(model, feature_names)
    
    print("\n[FIN] Résultats sauvegardés dans 'reports/model_results'.")

if __name__ == "__main__":
    main()