import glob
import pandas as pd

# 1

print("Reading CSV files...")
csv_files = glob.glob("data/*.csv")  # ou ton chemin exact
df_list = [pd.read_csv(file, sep=";", encoding="latin1") for file in csv_files]
df = pd.concat(df_list, ignore_index=True)

# 2 

import numpy as np

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

print("Building player stats...")
df_train = df[df['Date'] < '2025-01-01']
df_test = df[df['Date'] >= '2025-01-01']
player_stats = build_player_stats(df_train)

# 2.1 Head to head

def get_head_to_head(df):
    print("Computing head-to-head...")
    h2h = {}

    h2h_diff = []

    for _, row in df.sort_values('Date').iterrows():
        A, B = row['Winner'], row['Loser']
        key = tuple(sorted([A, B]))

        wins_A = h2h.get((A, B), 0)
        wins_B = h2h.get((B, A), 0)
        total = wins_A + wins_B if wins_A + wins_B > 0 else 1

        h2h_diff.append(wins_A / total - wins_B / total)

        h2h[(A, B)] = h2h.get((A, B), 0) + 1

    df['h2h_diff'] = h2h_diff
    return df

df = get_head_to_head(df)

# 3 
def build_match_dataset(df, player_stats):
    records = []

    for _, row in df.iterrows():
        A = row['Winner']
        B = row['Loser']
        surface = row['Surface']
        h2h = row.get('h2h_diff', 0)
        
        # Encoding du niveau de tournoi et du round
        tournament_level_map = {
            'G': 5,   # Grand Slam
            'M': 4,   # Masters 1000
            'A': 3,   # ATP 500
            'B': 2,   # ATP 250
            'C': 1    # Challenger
        }

        round_map = {
            'R128': 1, 'R64': 2, 'R32': 3, 'R16': 4,
            'QF': 5, 'SF': 6, 'F': 7
        }

        level = tournament_level_map.get(row.get('Tournament Level', 'C'), 1)
        round_num = round_map.get(row.get('Round', 'R32'), 3)


        if A not in player_stats['Player'].values or B not in player_stats['Player'].values:
            continue

        stats_A = player_stats[player_stats['Player'] == A].iloc[0]
        stats_B = player_stats[player_stats['Player'] == B].iloc[0]

        # Construire les features
        record = {
            'elo_diff': stats_A['elo'] - stats_B['elo'],
            'elo_surface_diff': stats_A.get(f'elo_{surface.lower()}', 1500) - stats_B.get(f'elo_{surface.lower()}', 1500),
            'winrate_diff': stats_A['winrate'] - stats_B['winrate'],
            'surface_winrate_diff': stats_A.get(f'mean_{surface}', 0) - stats_B.get(f'mean_{surface}', 0),
            'recent_form_diff': stats_A['recent_form'] - stats_B['recent_form'],
            'h2h-diff': h2h,
            'target': 1
        }
        records.append(record)

        # Miroir (match inversé)
        record_mirror = {
            'elo_diff': -record['elo_diff'],
            'elo_surface_diff': -record['elo_surface_diff'],
            'winrate_diff': -record['winrate_diff'],
            'surface_winrate_diff': -record['surface_winrate_diff'],
            'recent_form_diff': -record['recent_form_diff'],
            'h2h-diff': -record['h2h-diff'],
            'target': 0
        }
        records.append(record_mirror)

    return pd.DataFrame(records)


print("Building match dataset...")
match_df = build_match_dataset(df, player_stats)

# 4

from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report

print("Training model with XGBoost...")

match_train = build_match_dataset(df_train, player_stats)
match_test = build_match_dataset(df_test, player_stats)

X_train = match_train.drop(columns=['target'])
y_train = match_train['target']

model = LGBMClassifier(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

X_test = match_test.drop(columns=['target'])
y_test = match_test['target']

y_pred = model.predict(X_test)
print("✅ Accuracy sur 2025:", accuracy_score(y_test, y_pred))
print("📊 Report:\n", classification_report(y_test, y_pred))


# 5
def predict_match(playerA, playerB, surface, coteA, coteB):
    surface_key = f'elo_{surface.lower()}'
    mean_surface_key = f'mean_{surface}'

    if playerA not in player_stats['Player'].values or playerB not in player_stats['Player'].values:
        print(f"❌ Joueur manquant dans les stats.")
        return
    
    statsA = player_stats[player_stats['Player'] == playerA].iloc[0]
    statsB = player_stats[player_stats['Player'] == playerB].iloc[0]

    print(statsA)
    print(statsB)

    # Historique H2H
    past_matches = df[
        ((df['Winner'] == playerA) & (df['Loser'] == playerB)) |
        ((df['Winner'] == playerB) & (df['Loser'] == playerA))
    ]
    wins_A = past_matches[past_matches['Winner'] == playerA].shape[0]
    wins_B = past_matches[past_matches['Winner'] == playerB].shape[0]
    total = wins_A + wins_B if wins_A + wins_B > 0 else 1
    h2h_diff = (wins_A / total) - (wins_B / total)

    print("Head to Head diff : ", h2h_diff)

    match = pd.DataFrame([{
        'elo_diff': statsA['elo'] - statsB['elo'],
        'elo_surface_diff': statsA.get(surface_key, 1500) - statsB.get(surface_key, 1500),
        'winrate_diff': statsA['winrate'] - statsB['winrate'],
        'surface_winrate_diff': statsA.get(mean_surface_key, 0) - statsB.get(mean_surface_key, 0),
        'recent_form_diff': statsA['recent_form'] - statsB['recent_form'],
        'h2h-diff': h2h_diff
    }])

    proba = model.predict_proba(match)[0][1]
    print(f"\n🎾 {playerA} vs {playerB} on {surface}")
    print(f"➡️ Proba que {playerA} gagne : {proba:.2%}")
    if (proba > 0.5):
        expectedValue = (proba * coteA) - 1
    else:
        expectedValue = (proba * coteB) - 1
    print("Expected Value = ", expectedValue) # On veut que ce soit > 0


# tournament_level_map = {
#             'G': 5,   # Grand Slam
#             'M': 4,   # Masters 1000
#             'A': 3,   # ATP 500
#             'B': 2,   # ATP 250
#             'C': 1    # Challenger
#         }

#         round_map = {
#             'R128': 1, 'R64': 2, 'R32': 3, 'R16': 4,
#             'QF': 5, 'SF': 6, 'F': 7
#         }
