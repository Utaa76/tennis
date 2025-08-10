# Amélioration complète du prédicteur de matchs de tennis
# - Séparation temporelle train/test
# - Modèle XGBoost
# - Ajout de nouvelles features
# - Normalisation des données
# - GridSearch pour optimisation

import glob
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 1. Lecture des données
csv_files = glob.glob("data/*.csv")
df_list = [pd.read_csv(file, sep=";", encoding="latin1") for file in csv_files]
df = pd.concat(df_list, ignore_index=True)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

# 2. Séparation temporelle
train_df = df[df['Date'].dt.year < 2025].copy()
test_df = df[df['Date'].dt.year == 2025].copy()

# 3. Construction des stats joueur (comme avant, à factoriser dans une fonction build_player_stats)
def build_player_stats(df):
    df = df.copy()
    all_matches = []

    for index, row in df.iterrows():
        winner = row['Winner']
        loser = row['Loser']
        surface = row['Surface']
        rank_w, rank_l = row.get('WRank', np.nan), row.get('LRank', np.nan)

        all_matches.append({'Player': winner, 'result': 1, 'surface': surface, 'rank': rank_w, 'date': row['Date']})
        all_matches.append({'Player': loser, 'result': 0, 'surface': surface, 'rank': rank_l, 'date': row['Date']})

    matches_df = pd.DataFrame(all_matches)
    matches_df['date'] = pd.to_datetime(matches_df['date'], format="%d/%m/%Y", errors='coerce')

    # Elo simple
    def compute_elo(df_matches, k=32):
        print("Computing elo...")
        elo = {}
        elo_history = []

        for _, row in df_matches.sort_values('date').iterrows():
            player = row['Player']
            result = row['result']

            player_elo = elo.get(player, 1500)
            opponent_row = df_matches[(df_matches['date'] == row['date']) & (df_matches['Player'] != player)]
            if opponent_row.empty:
                continue
            opponent = opponent_row.iloc[0]['Player']
            opponent_elo = elo.get(opponent, 1500)

            expected = 1 / (1 + 10 ** ((opponent_elo - player_elo) / 400))
            new_elo = player_elo + k * (result - expected)
            elo[player] = new_elo

            elo_history.append({'Player': player, 'elo': new_elo})

        return pd.DataFrame(elo_history).groupby('Player')['elo'].last().reset_index()
    
    def compute_elo_surface(df_matches, k=32):
        print("Computing elo by surface...")
        elo = {}  # (player, surface) → elo
        elo_history = []

        for _, row in df_matches.sort_values('date').iterrows():
            player = row['Player']
            result = row['result']
            surface = row['surface']

            # Elo actuel du joueur sur cette surface
            player_elo = elo.get((player, surface), 1500)

            # Trouver l’adversaire dans le même match
            opponent_row = df_matches[
                (df_matches['date'] == row['date']) &
                (df_matches['Player'] != player) &
                (df_matches['surface'] == surface)
            ]
            if opponent_row.empty:
                continue

            opponent = opponent_row.iloc[0]['Player']
            opponent_elo = elo.get((opponent, surface), 1500)

            expected = 1 / (1 + 10 ** ((opponent_elo - player_elo) / 400))
            new_elo = player_elo + k * (result - expected)
            elo[(player, surface)] = new_elo

            elo_history.append({
                'Player': player,
                'surface': surface,
                'elo': new_elo
            })

        # Regroupement final par joueur et surface
        df_elo = pd.DataFrame(elo_history)
        df_elo = df_elo.groupby(['Player', 'surface'])['elo'].last().unstack().reset_index()

        # Renommer les colonnes selon surface
        surface_column_map = {
            'Hard': 'elo_hard',
            'Clay': 'elo_clay',
            'Grass': 'elo_grass',
            'Carpet': 'elo_carpet'
        }
        df_elo = df_elo.rename(columns=surface_column_map)

        return df_elo


    
    def compute_recent_form(matches_df, n=10):
        print("Computing recent form...")
        recent_form = []

        # On trie par joueur et date pour garder l’ordre chronologique
        matches_df = matches_df.sort_values(['Player', 'date'])

        players = matches_df['Player'].unique()

        for player in players:
            player_matches = matches_df[matches_df['Player'] == player]

            # On récupère les n derniers résultats (victoire=1, défaite=0)
            last_n_results = player_matches['result'].tail(n)

            if len(last_n_results) == 0:
                form = None
            else:
                form = last_n_results.mean()

            recent_form.append({'Player': player, 'recent_form': form})

        return pd.DataFrame(recent_form)

    elo_df = compute_elo(matches_df)
    elo_surface_df = compute_elo_surface(matches_df)
    recent_form_df = compute_recent_form(matches_df)

    # Winrates & surfaces
    stats = matches_df.groupby('Player').agg(
        matches_played=('result', 'count'),
        wins=('result', 'sum'),
        winrate=('result', 'mean'),
        avg_rank=('rank', 'mean')
    ).reset_index()

    for surf in ['Clay', 'Grass', 'Hard']:
        sub = matches_df[matches_df['surface'] == surf]
        surface_counts = sub.groupby('Player')['result'].count()
        surface_winrates = sub.groupby('Player')['result'].mean()

        stats[f'count_{surf}'] = stats['Player'].map(surface_counts).fillna(0)
        stats[f'mean_{surf}'] = stats['Player'].map(surface_winrates).fillna(0)

    # Fusion progressive
    full_stats = stats.merge(elo_df, on='Player', how='left')
    full_stats = full_stats.merge(elo_surface_df, on='Player', how='left')
    full_stats = full_stats.merge(recent_form_df, on='Player', how='left')
    full_stats = full_stats.fillna(0)

    return full_stats

player_stats = build_player_stats(train_df)

# 4. Fonction de build match avec features améliorées

def build_match_dataset(df, player_stats):
    records = []
    for _, row in df.iterrows():
        if row['Winner'] not in player_stats['Player'].values or row['Loser'] not in player_stats['Player'].values:
            continue

        A = row['Winner']
        B = row['Loser']
        surface = row['Surface']

        stats_A = player_stats[player_stats['Player'] == A].iloc[0]
        stats_B = player_stats[player_stats['Player'] == B].iloc[0]

        record = {
            'elo_diff': stats_A['elo'] - stats_B['elo'],
            'elo_surface_diff': stats_A.get(f'elo_{surface.lower()}', 1500) - stats_B.get(f'elo_{surface.lower()}', 1500),
            'winrate_diff': stats_A['winrate'] - stats_B['winrate'],
            'surface_winrate_diff': stats_A.get(f'mean_{surface}', 0) - stats_B.get(f'mean_{surface}', 0),
            'recent_form_diff': stats_A['recent_form'] - stats_B['recent_form'],
            'ranking_diff': stats_B['avg_rank'] - stats_A['avg_rank'],
            'target': 1
        }
        records.append(record)

        # Miroir (match inversé)
        record_mirror = {k: -v if k != 'target' else 0 for k, v in record.items()}
        records.append(record_mirror)

    return pd.DataFrame(records)

train_match_df = build_match_dataset(train_df, player_stats)
test_match_df = build_match_dataset(test_df, player_stats)

# 5. Données d'entraînement
X_train = train_match_df.drop(columns='target')
y_train = train_match_df['target']
X_test = test_match_df.drop(columns='target')
y_test = test_match_df['target']

# 6. Modèle avec grid search XGBoost
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])

param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [3, 6],
    'model__learning_rate': [0.05, 0.1]
}

grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', verbose=1)
grid.fit(X_train, y_train)

# 7. Évaluation
print("\nBest params:", grid.best_params_)
y_pred = grid.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 8. Prédiction personnalisée
model = grid.best_estimator_

def predict_match(playerA, playerB, surface):
    if playerA not in player_stats['Player'].values or playerB not in player_stats['Player'].values:
        print("Joueur manquant dans les stats.")
        return

    statsA = player_stats[player_stats['Player'] == playerA].iloc[0]
    statsB = player_stats[player_stats['Player'] == playerB].iloc[0]

    surface_key = f'elo_{surface.lower()}'
    mean_surface_key = f'mean_{surface}'

    match = pd.DataFrame([{        
        'elo_diff': statsA['elo'] - statsB['elo'],
        'elo_surface_diff': statsA.get(surface_key, 1500) - statsB.get(surface_key, 1500),
        'winrate_diff': statsA['winrate'] - statsB['winrate'],
        'surface_winrate_diff': statsA.get(mean_surface_key, 0) - statsB.get(mean_surface_key, 0),
        'recent_form_diff': statsA['recent_form'] - statsB['recent_form'],
        'ranking_diff': statsB['avg_rank'] - statsA['avg_rank']
    }])

    proba = model.predict_proba(match)[0][1]
    print(f"\n🎾 {playerA} vs {playerB} on {surface}")
    print(f"➡️ Proba que {playerA} gagne : {proba:.2%}")
