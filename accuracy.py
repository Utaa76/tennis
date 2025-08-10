import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# === 1. Charger le CSV ===
df = pd.read_csv("2024.csv", sep=";")

print(f"✅ Nombre de matchs avant nettoyage : {len(df)}")

# === 2. Nettoyage de base ===
# Garder uniquement les lignes complètes
colonnes_utiles = ['Surface', 'WRank', 'LRank', 'B365W', 'B365L', 'Winner', 'Loser']
df = df.dropna(subset=colonnes_utiles)

# Changer les valeurs décimales à virgules en valeurs décimales à points
df['B365W'] = df['B365W'].str.replace(',', '.')
df['B365L'] = df['B365L'].str.replace(',', '.')

# Convertir en numérique
for col in ['WRank', 'LRank', 'B365W', 'B365L']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Supprimer les lignes encore corrompues
df = df.dropna(subset=['WRank', 'LRank', 'B365W', 'B365L'])
    
print("NaN dans B365W :", df['B365W'].isna().sum())
print("NaN dans B365L :", df['B365L'].isna().sum())

# Supprimer les lignes encore corrompues
print(f"✅ Nombre de matchs après nettoyage : {len(df)}")
print(df[['WRank', 'LRank', 'B365W', 'B365L']].describe())

# === 3. Feature engineering ===
df['rank_diff'] = df['LRank'] - df['WRank']       # écart de classement
df['odds_diff'] = df['B365L'] - df['B365W']       # écart de cotes
df['target'] = 1                                  # le joueur A (Winner) est toujours gagnant ici

df_win = df.copy()
df_win['target'] = 1

df_lose = df.copy()
df_lose[['WRank', 'LRank']] = df_lose[['LRank', 'WRank']]
df_lose[['B365W', 'B365L']] = df_lose[['B365L', 'B365W']]
df_lose['target'] = 0

df_full = pd.concat([df_win, df_lose], ignore_index=True)

# === 4. Construire X et y ===
X = df_full[['rank_diff', 'odds_diff']]
y = df_full['target']

# recalculer rank_diff et odds_diff dans df_full
df_full['rank_diff'] = df_full['LRank'] - df_full['WRank']
df_full['odds_diff'] = df_full['B365L'] - df_full['B365W']

# === 5. Split et entraînement ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# === 6. Évaluation ===
y_pred = model.predict(X_test)

print("\n✅ Accuracy du modèle :", round(accuracy_score(y_test, y_pred), 3))
print("\n📊 Rapport de classification :\n", classification_report(y_test, y_pred))

# === 7. Exemple de prédiction manuelle ===
# Exemple : joueur A classé 20, joueur B classé 35, cotes 1.70 vs 2.20
import numpy as np

match = pd.DataFrame([{
    'rank_diff': 35 - 20,
    'odds_diff': 2.20 - 1.70
}])

proba = model.predict_proba(match)[0][1]
print(f"\n🔮 Probabilité que le joueur A gagne : {proba:.2%}")
