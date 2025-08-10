import glob
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from scipy.special import logit, expit
from scipy.optimize import minimize
from collections import defaultdict, deque
import random

# 1 - Chargement des données CSV
print("Reading CSV files...")
csv_files = glob.glob("data/*.csv")  # adapte ton chemin
df_list = [pd.read_csv(file, sep=";", encoding="latin1") for file in csv_files]
df = pd.concat(df_list, ignore_index=True)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)

# 2 - Fonction de construction du dataset (copié du LightGBM)
def build_match_dataset_online(df_matches, k_elo=32, recent_n=50):
    elo = defaultdict(lambda: 1500.)
    elo_surf = defaultdict(lambda: 1500.)
    wins = defaultdict(int)
    games = defaultdict(int)
    surface_win = defaultdict(int)
    surface_games = defaultdict(int)
    recent = defaultdict(lambda: deque(maxlen=recent_n))
    h2h = defaultdict(int)
    last_rank = {}
    last_points = {}
    recent_games_diff = defaultdict(lambda:deque(maxlen=recent_n))
    recent_decider = defaultdict(lambda:deque(maxlen=recent_n))

    records = []
    tournament_level_map = {
        "Grand Slam": 4,
        "Masters 1000": 3,
        "ATP500": 2,
        "ATP250": 1,
    }
    round_map = {
        '1st Round': 1,
        '2nd Round': 2,
        '3rd Round': 3,
        '4th Round': 4,
        'Quarterfinals': 5,
        'Semifinals': 6,
        'The Final': 7
    }

    for _, row in df_matches.sort_values('Date').iterrows():
        if random.random() < 0.5:
            A, B = row['Winner'], row['Loser']
            rank_a, rank_b = row['WRank'], row['LRank']
            pts_a, pts_b = row['WPts'], row['LPts']
            target = 1
            ow = float(str(row.get('B365W', np.nan)).replace(',', '.'))
            ol = float(str(row.get('B365L', np.nan)).replace(',', '.'))
        else:
            A, B = row['Loser'], row['Winner']
            rank_a, rank_b = row['LRank'], row['WRank']
            pts_a, pts_b = row['LPts'], row['WPts']
            target = 0
            ow = float(str(row.get('B365L', np.nan)).replace(',', '.'))
            ol = float(str(row.get('B365W', np.nan)).replace(',', '.'))

        surface = row['Surface'].lower()
        date = row['Date']
        if A not in last_rank or last_rank[A][0] < date:
            last_rank[A] = (date, rank_a)
            last_points[A] = (date, pts_a)
        if B not in last_rank or last_rank[B][0] < date:
            last_rank[B] = (date, rank_b)
            last_points[B] = (date, pts_b)

        feat = {}
        feat['elo_diff'] = elo[A] - elo[B]
        feat['elo_surface_diff'] = (elo_surf[(A,surface)] - elo_surf[(B,surface)])

        wr_A = wins[A] / games[A] if games[A] else 0
        wr_B = wins[B] / games[B] if games[B] else 0
        feat['winrate_diff'] = wr_A - wr_B

        swr_A = (surface_win[(A,surface)] / surface_games[(A,surface)] if surface_games[(A,surface)] else 0)
        swr_B = (surface_win[(B,surface)] / surface_games[(B,surface)] if surface_games[(B,surface)] else 0)
        feat['surface_winrate_diff'] = swr_A - swr_B

        form_A = np.mean(recent[A]) if recent[A] else 0.5
        form_B = np.mean(recent[B]) if recent[B] else 0.5
        feat['recent_form_diff'] = form_A - form_B

        total_h2h = h2h[(A,B)] + h2h[(B,A)]
        h2h_diff = (h2h[(A,B)] - h2h[(B,A)]) / total_h2h if total_h2h else 0
        feat['h2h_diff'] = h2h_diff

        feat['odds_ratio'] = np.log(ol / ow) if ow > 0 and ol > 0 else 0
        feat['elo_surface'] = feat['elo_diff'] * feat['elo_surface_diff']

        feat['level'] = tournament_level_map.get(row.get('Series', 'ATP250'), 1)
        feat['round'] = round_map.get(row.get('Round','1st Round'),1)
        feat['OddW'] = ow
        feat['OddL'] = ol
        feat['rank_diff'] = rank_a - rank_b
        feat['pts_diff'] = pts_a - pts_b

        set_cols_w = [f'W{i}' for i in range(1, 6) if f'W{i}' in row]
        set_cols_l = [f'L{i}' for i in range(1, 6) if f'L{i}' in row]

        games_w = sum(row[c] for c in set_cols_w if not pd.isna(row[c]))
        games_l = sum(row[c] for c in set_cols_l if not pd.isna(row[c]))
        nb_sets = len([c for c in set_cols_w if not pd.isna(row[c])])

        tiebreaks_w = sum(1 for c in set_cols_w if row[c] == 7 and row[c.replace('W', 'L')] > 5)
        tiebreaks_l = sum(1 for c in set_cols_l if row[c] == 7 and row[c.replace('L', 'W')] > 5)

        rolling_n = recent_n
        gd_A = np.mean(list(recent_games_diff[A])[-rolling_n:]) if recent_games_diff[A] else 0
        gd_B = np.mean(list(recent_games_diff[B])[-rolling_n:]) if recent_games_diff[B] else 0
        feat['games_diff_recent'] = gd_A - gd_B

        decider_A = np.mean(recent_decider[A]) if recent_decider[A] else 0.5
        decider_B = np.mean(recent_decider[B]) if recent_decider[B] else 0.5
        feat['decider_winrate_diff'] = decider_A - decider_B

        feat['target'] = target
        feat['Winner'] = A
        feat['Loser'] = B
        feat['Date'] = date
        feat['Surface'] = surface
        records.append(feat)

        result_A = target
        result_B = 1 - target

        expected_A = 1 / (1 + 10 ** ((elo[B] - elo[A]) / 400))
        elo[A] += k_elo * (result_A - expected_A)
        elo[B] += k_elo * (result_B - (1 - expected_A))

        expected_surf_A = 1 / (1 + 10 ** ((elo_surf[(B, surface)] - elo_surf[(A, surface)]) / 400))
        elo_surf[(A, surface)] += k_elo * (result_A - expected_surf_A)
        elo_surf[(B, surface)] += k_elo * (result_B - (1 - expected_surf_A))

        wins[A] += result_A
        wins[B] += result_B
        games[A] += 1
        games[B] += 1

        surface_win[(A, surface)] += result_A
        surface_win[(B, surface)] += result_B
        surface_games[(A, surface)] += 1
        surface_games[(B, surface)] += 1

        if target == 1:
            recent_games_diff[A].append(games_w - games_l)
            recent_games_diff[B].append(games_l - games_w)
            recent_decider[A].append(int((nb_sets == 3 or nb_sets == 5) and target == 1))
            recent_decider[B].append(int((nb_sets == 3 or nb_sets == 5) and target == 0))
        else:
            recent_games_diff[B].append(games_w - games_l)
            recent_games_diff[A].append(games_l - games_w)
            recent_decider[B].append(int((nb_sets == 3 or nb_sets == 5) and target == 1))
            recent_decider[A].append(int((nb_sets == 3 or nb_sets == 5) and target == 0))

        recent[A].append(result_A)
        recent[B].append(result_B)
        h2h[(A, B)] += 1

    dataset = pd.DataFrame(records)
    return dataset

print("Building match dataset...")
df = df.sort_values(by="Date")
match_df = build_match_dataset_online(df)

# 3 - Préparation train/test
match_train = match_df[match_df['Date'] < '2025-01-01']
match_test = match_df[match_df['Date'] >= '2025-01-01']

X_train = match_train.drop(columns=['Winner', 'Loser', 'target', 'Date', 'Surface'])
y_train = match_train['target']
X_test = match_test.drop(columns=['Winner', 'Loser', 'target', 'Date', 'Surface'])
y_test = match_test['target']

# 4 - Entraînement SGDClassifier
from sklearn.impute import SimpleImputer

# Imputation des NaN par la moyenne sur chaque colonne
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Maintenant on entraîne avec les données imputées
sgd = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42)
sgd.fit(X_train_imputed, y_train)

# Pour la prédiction
y_pred = sgd.predict(X_test_imputed)

# 5 - Évaluation basique
y_pred = sgd.predict(X_test_imputed)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy sur données de test (2025): {acc:.4f}")
print("📊 Rapport de classification :")
print(classification_report(y_test, y_pred))

import joblib
joblib.dump(sgd, "modeleSGD_tennis.pkl")