import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Exemple simplifié avec 10 matchs
data = {
    'rank_A': [1, 5, 10, 3, 7, 20, 15, 8, 2, 4],
    'rank_B': [2, 8, 5, 12, 6, 18, 30, 6, 7, 1],
    'win_rate_A': [0.9, 0.7, 0.6, 0.8, 0.65, 0.5, 0.55, 0.6, 0.85, 0.75],
    'win_rate_B': [0.85, 0.6, 0.7, 0.5, 0.68, 0.4, 0.45, 0.65, 0.7, 0.9],
    'surface': ['Hard', 'Clay', 'Hard', 'Grass', 'Hard', 'Clay', 'Clay', 'Hard', 'Grass', 'Hard'],
    'winner': [1, 1, 0, 1, 0, 1, 0, 1, 1, 0],  # 1 = joueur A gagne, 0 = joueur B gagne
}
df = pd.DataFrame(data)

print(df)

# Encodage de la surface (une-hot encoding)
df = pd.get_dummies(df, columns=['surface'])

# Création de features différentielles
df['rank_diff'] = df['rank_B'] - df['rank_A']
df['win_rate_diff'] = df['win_rate_A'] - df['win_rate_B']

# Variables explicatives (X) et cible (y)
X = df[['rank_diff', 'win_rate_diff', 'surface_Clay', 'surface_Grass', 'surface_Hard']]
y = df['winner']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modèle
model = LogisticRegression()
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")