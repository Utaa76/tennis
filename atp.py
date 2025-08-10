import pandas as pd
import glob

# Lire tous les CSV du dossier "data"
csv_files = glob.glob("data/*.csv")

df_list = []

for file in csv_files:
    try:
        df_temp = pd.read_csv(file, sep=";", encoding="utf-8")
        print(f"✅ {file} lu avec succès (UTF-8)")
    except UnicodeDecodeError:
        try:
            df_temp = pd.read_csv(file, sep=";", encoding="latin1")
            print(f"✅ {file} lu avec succès (Latin1)")
        except Exception as e:
            print(f"❌ Erreur de lecture pour {file} : {e}")
            continue  # skip le fichier si aucune lecture ne fonctionne

    df_list.append(df_temp)

# Concaténer tous les DataFrames en un seul
df = pd.concat(df_list, ignore_index=True)
# print(f"\n📊 Total de lignes après concaténation : {len(df)}")

for col in ['B365W', 'B365L', 'PSW', 'PSL', 'MaxW', 'MaxL', 'AvgW', 'AvgL']:
    df[col] = df[col].str.replace(',', '.')

# Convertir les classements et les cotes en float
df['WRank'] = pd.to_numeric(df['WRank'], errors='coerce')
df['LRank'] = pd.to_numeric(df['LRank'], errors='coerce')
df['B365W'] = pd.to_numeric(df['B365W'], errors='coerce')
df['B365L'] = pd.to_numeric(df['B365L'], errors='coerce')

# Supprimer les lignes restantes mal formatées
df = df.dropna(subset=['WRank', 'LRank', 'B365W', 'B365L'])

# Extrait des colonnes utiles
df_winner = df[['Winner', 'Surface', 'WRank', 'Date']].copy()
df_winner['Result'] = 1  # Victoire
df_winner.rename(columns={'Winner': 'Player', 'WRank': 'Rank'}, inplace=True)

df_loser = df[['Loser', 'Surface', 'LRank', 'Date']].copy()
df_loser['Result'] = 0  # Défaite
df_loser.rename(columns={'Loser': 'Player', 'LRank': 'Rank'}, inplace=True)

df_long = pd.concat([df_winner, df_loser], ignore_index=True)

# Statistiques globales

stats_global = df_long.groupby('Player').agg(
    matches_played=('Result', 'count'),
    wins=('Result', 'sum'),
    winrate=('Result', 'mean'),
    avg_rank=('Rank', 'mean')
).reset_index()

# print(stats_global.sort_values(by='avg_rank'))

# Statistiques par surface 

stats_surface = df_long.pivot_table(
    index='Player',
    columns='Surface',
    values='Result',
    aggfunc=['count', 'mean']  # count = nb matchs, mean = taux victoire
)

# Simplifier les colonnes (niveau multi-index)
stats_surface.columns = ['_'.join(col).strip() for col in stats_surface.columns.values]

stats_surface = stats_surface.reset_index()

player_stats = stats_global.merge(stats_surface, on='Player', how='left')

# print(player_stats)


# Ajout de l'élo

def update_elo(winner, loser, k=32):
    Ra = elo.get(winner, INITIAL_ELO)
    Rb = elo.get(loser, INITIAL_ELO)
    
    Ea = 1 / (1 + 10 ** ((Rb - Ra) / 400))  # probabilité que le winner gagne
    Eb = 1 - Ea
    
    elo[winner] = Ra + k * (1 - Ea)
    elo[loser]  = Rb + k * (0 - Eb)

INITIAL_ELO = 1500
elo_dict = {}
elo_history = {}

# Listes pour stockage temporaire
elo_winner_list = []
elo_loser_list = []

# Parcours de tous les matchs
for _, row in df.iterrows():
    winner = row['Winner']
    loser = row['Loser']
    
    # Obtenir ou initialiser les Elos
    elo_w = elo_dict.get(winner, INITIAL_ELO)
    elo_l = elo_dict.get(loser, INITIAL_ELO)
    
    # Stocker l'Elo avant match
    elo_winner_list.append(elo_w)
    elo_loser_list.append(elo_l)
    
    # Stocker pour l’historique par joueur
    for p, elo in [(winner, elo_w), (loser, elo_l)]:
        if p not in elo_history:
            elo_history[p] = []
        elo_history[p].append(elo)
    
    # Calcul de probabilité
    expected_w = 1 / (1 + 10 ** ((elo_l - elo_w) / 400))
    
    # Mise à jour des Elos
    K = 32
    elo_dict[winner] = elo_w + K * (1 - expected_w)
    elo_dict[loser] = elo_l + K * (0 - (1 - expected_w))
    
elo_final = pd.DataFrame([
    {'Player': player, 'elo': elo_dict[player]} for player in elo_dict
])

player_stats = player_stats.merge(elo_final, how='left', left_on='Player', right_on='Player')

# print(player_stats.sort_values(by='elo').tail(20))

######################################

import pandas as pd

# Helper : récupérer les features d’un joueur
def get_player_features(player_name):
    row = player_stats[player_stats['Player'] == player_name]
    return row.squeeze() if not row.empty else None

# Encodage des surfaces
surface_map = {"Hard": 0, "Clay": 1, "Grass": 2}

features = []

for _, row in df.iterrows():
    winner = row["Winner"]
    loser = row["Loser"]
    surface = row["Surface"]

    p1 = get_player_features(winner)
    p2 = get_player_features(loser)

    if p1 is None or p2 is None or surface not in surface_map:
        continue

    surface_code = surface_map[surface]
    surface_col = f"mean_{surface}"

    # S'assurer que les colonnes existent
    if surface_col not in p1 or surface_col not in p2:
        continue

    feature_row = {
        "elo_diff": p1["elo"] - p2["elo"],
        "winrate_diff": p1["winrate"] - p2["winrate"],
        "avg_rank_diff": p2["avg_rank"] - p1["avg_rank"],  # classement inversé
        "surface": surface_code,
        "surface_winrate_diff": p1[surface_col] - p2[surface_col],
        "target": 1  # le joueur A (Winner) gagne
    }

    features.append(feature_row)

# Création du DataFrame final
# Surface mapping
surface_map = {"Clay": 0, "Grass": 1, "Hard": 2}

# === 1. Matchs "positifs" (le joueur A a gagné)
pos_matches = df.copy()
pos_matches["PlayerA"] = pos_matches["Winner"]
pos_matches["PlayerB"] = pos_matches["Loser"]
pos_matches["target"] = 1
pos_matches["surface_code"] = pos_matches["Surface"].map(surface_map)

# === 2. Matchs "négatifs" (inversion des joueurs)
neg_matches = df.copy()
neg_matches["PlayerA"] = neg_matches["Loser"]
neg_matches["PlayerB"] = neg_matches["Winner"]
neg_matches["target"] = 0
neg_matches["surface_code"] = neg_matches["Surface"].map(surface_map)

# === 3. Concaténer les deux
full_matches = pd.concat([pos_matches, neg_matches], ignore_index=True)

# === 4. Créer les features comme précédemment
def get_player_features(name):
    row = player_stats[player_stats["Player"] == name]
    if row.empty:
        return None
    return row.iloc[0]

match_rows = []
for _, row in full_matches.iterrows():
    p1 = get_player_features(row["PlayerA"])
    p2 = get_player_features(row["PlayerB"])
    if p1 is None or p2 is None or pd.isna(row["surface_code"]):
        continue
    match_rows.append({
        "elo_diff": p1["elo"] - p2["elo"],
        "winrate_diff": p1["winrate"] - p2["winrate"],
        "avg_rank_diff": p2["avg_rank"] - p1["avg_rank"],
        "surface": row["surface_code"],
        "surface_winrate_diff": p1[f"mean_{row['Surface']}"] - p2[f"mean_{row['Surface']}"],
        "target": row["target"]
    })

match_df = pd.DataFrame(match_rows)
print("✅ Nombre de matchs utilisés pour le modèle :", len(match_df))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# === 1. Préparation des données ===
X = match_df.drop(columns=["target"])
y = match_df["target"]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 2. Modèle ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === 3. Évaluation ===
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy : {accuracy:.3f}")
print("\n📊 Rapport de classification :\n", classification_report(y_test, y_pred))

# === 4. Exemple de prédiction manuelle ===
def predict_match(player_a, player_b, surface):
    p1 = get_player_features(player_a)
    p2 = get_player_features(player_b)
    surface_code = surface_map.get(surface, -1)
    surface_col = f"mean_{surface}"

    if p1 is None or p2 is None or surface_code == -1:
        print("❌ Données manquantes pour l’un des joueurs ou surface inconnue")
        return

    match_features = pd.DataFrame([{
        "elo_diff": p1["elo"] - p2["elo"],
        "winrate_diff": p1["winrate"] - p2["winrate"],
        "avg_rank_diff": p2["avg_rank"] - p1["avg_rank"],
        "surface": surface_code,
        "surface_winrate_diff": p1[surface_col] - p2[surface_col],
    }])

    proba = model.predict_proba(match_features)[0][1]
    print(f"\n🔮 Probabilité que {player_a} batte {player_b} sur {surface} : {proba:.2%}")

# Exemple : prédire Djokovic vs Alcaraz sur Clay
predict_match("Sonego L.", "Nakashima B.", "Grass")