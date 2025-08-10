import glob
import pandas as pd
from scipy.special import logit, expit          # logit/σ
from scipy.optimize import minimize            # descente Nelder‑Mead
import joblib

seed = 20
modelLGBM = joblib.load("modeleLGBM_tennis_2024-20.pkl")

# 1

print("Reading CSV files...")
csv_files = glob.glob("data/*.csv")  # ou ton chemin exact
df_list = [pd.read_csv(file, sep=";", encoding="latin1") for file in csv_files]
df = pd.concat(df_list, ignore_index=True)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)

# 2 

import numpy as np

import joblib

# model = joblib.load("modele_tennis.pkl")
# df = joblib.load("df_tennis.pkl")
df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
# player_stats = joblib.load("players_stats.pkl")

df_train = df[df['Date'] < '2025-01-01']
df_test = df[df['Date'] >= '2025-01-01']

# 3 
from collections import defaultdict, deque
import random

SURFACES = ['clay', 'grass', 'hard', 'carpet']

def default_stats(p):
    return {
        'elo': 1500, 'winrate': 0, 'wrlast50': 0.5, 'wrlast100': 0.5, 'rank': 999, 'pts': 0,
        'recent_games_diff_last': 0, 'recent_decider_last': 0.5,
        **{f'elo_{s}': 1500 for s in SURFACES},
        **{f'mean_{s}': 0 for s in SURFACES}
    }

def build_match_dataset_online(df_matches, k_elo=32, recent_n=50, seed=42):
    df_matches = df_matches.sort_values('Date')
    player_stats_dict = {}

    wins = defaultdict(int)
    games = defaultdict(int)
    surface_win = defaultdict(int)
    surface_games = defaultdict(int)
    last50 = defaultdict(lambda: deque(maxlen=recent_n))
    last100 = defaultdict(lambda: deque(maxlen=recent_n))
    recent_games_diff = defaultdict(lambda: deque(maxlen=recent_n))
    recent_decider = defaultdict(lambda: deque(maxlen=recent_n))
    h2h = defaultdict(int)
    h2h_surface = defaultdict(int)  # nouveau : (A, B, surface) -> victoires

    tournament_level_map = {
        "Grand Slam": 4, "Masters 1000": 3, "ATP500": 2, "ATP250": 1,
        "WTA250": 1, "WTA500": 2, "WTA1000": 3
    }
    round_map = {
        '1st Round': 1, '2nd Round': 2, '3rd Round': 3, '4th Round': 4,
        'Quarterfinals': 5, 'Semifinals': 6, 'The Final': 7
    }

    records = []
    random.seed(seed)

    for _, row in df_matches.iterrows():
        print(row['Date'])
        A, B = row['Winner'], row['Loser']
        odda = float(str(row.get('B365W', np.nan)).replace(',', '.'))
        oddb = float(str(row.get('B365L', np.nan)).replace(',', '.'))
        rank_a, rank_b = row['WRank'], row['LRank']
        pts_a, pts_b = row['WPts'], row['LPts']
        target = 1

        if np.isnan(odda) or np.isnan(oddb):
            continue

        statsA = player_stats_dict.get(A, default_stats(A)).copy()
        statsB = player_stats_dict.get(B, default_stats(B)).copy()

        surface = row['Surface'].lower()
        date = row['Date']

        # H2H global
        h2h_A_vs_B = h2h[(A, B)]
        h2h_B_vs_A = h2h[(B, A)]
        total_h2h = h2h_A_vs_B + h2h_B_vs_A
        h2h_diff = ((h2h_A_vs_B - h2h_B_vs_A) / total_h2h) if total_h2h > 0 else 0

        # H2H par surface
        h2h_surf_A_vs_B = h2h_surface[(A, B, surface)]
        h2h_surf_B_vs_A = h2h_surface[(B, A, surface)]
        total_h2h_surf = h2h_surf_A_vs_B + h2h_surf_B_vs_A
        h2h_surface_diff = ((h2h_surf_A_vs_B - h2h_surf_B_vs_A) / total_h2h_surf) if total_h2h_surf > 0 else 0

        features = [
            'elo_diff', 
            'elo_surface_diff', 
            'winrate_diff', 
            'surface_winrate_diff',
            'wrlast50_diff',
            'wrlast100_diff',
            'h2h_diff', 
            'h2h_surface_diff', 
            'odds-ratio', 
            'level',
            'round',
            'rank_diff',
            'pts_diff',
            'games_diff_recent',
            'decider_winrate_diff'
        ]

        feat_A = {
            'elo_diff': statsA['elo'] - statsB['elo'],
            'elo_surface_diff': statsA[f'elo_{surface}'] - statsB[f'elo_{surface}'],
            'winrate_diff': statsA['winrate'] - statsB['winrate'],
            'surface_winrate_diff': statsA[f'mean_{surface}'] - statsB[f'mean_{surface}'],
            'wrlast50_diff': statsA['wrlast50'] - statsB['wrlast50'],
            'wrlast100_diff': statsA['wrlast100'] - statsB['wrlast100'],
            'h2h_diff': h2h_diff,
            'h2h_surface_diff': h2h_surface_diff,  # nouveau feature
            'odds-ratio': np.log(oddb / odda) * 5 if odda > 0 and oddb > 0 else 0,
            'level': tournament_level_map.get(row.get('Series', 'ATP250'), 1),
            'round': round_map.get(row.get('Round', '1st Round'), 1),
            'rank_diff': rank_a - rank_b,
            'pts_diff': pts_a - pts_b,
            'OddA': odda,
            'OddB': oddb,
            'games_diff_recent': statsA['recent_games_diff_last'] - statsB['recent_games_diff_last'],
            'decider_winrate_diff': statsA['recent_decider_last'] - statsB['recent_decider_last'],
            'target': 1,
            'PlayerA': A,
            'PlayerB': B,
            'Date': date,
            'Surface': surface,
            'Winner': row['Winner'],
            'Loser': row['Loser'],
            'gain_net': (odda - 1) if A == row['Winner'] else -1,
            'pari_sur': A
        }

        feat_B = {
            'elo_diff': statsB['elo'] - statsA['elo'],
            'elo_surface_diff': statsB[f'elo_{surface}'] - statsA[f'elo_{surface}'],
            'winrate_diff': statsB['winrate'] - statsA['winrate'],
            'surface_winrate_diff': statsB[f'mean_{surface}'] - statsA[f'mean_{surface}'],
            'wrlast50_diff': statsB['wrlast50'] - statsA['wrlast50'],
            'wrlast100_diff': statsB['wrlast100'] - statsA['wrlast100'],
            'h2h_diff': -h2h_diff,
            'h2h_surface_diff': -h2h_surface_diff,  # nouveau feature
            'odds-ratio': np.log(odda / oddb) * 5 if odda > 0 and oddb > 0 else 0,
            'level': tournament_level_map.get(row.get('Series', 'ATP250'), 1),
            'round': round_map.get(row.get('Round', '1st Round'), 1),
            'rank_diff': rank_b - rank_a,
            'pts_diff': pts_b - pts_a,
            'OddA': odda,
            'OddB': oddb,
            'games_diff_recent': statsB['recent_games_diff_last'] - statsA['recent_games_diff_last'],
            'decider_winrate_diff': statsB['recent_decider_last'] - statsA['recent_decider_last'],
            'target': 0,
            'PlayerA': A,
            'PlayerB': B,
            'Date': date,
            'Surface': surface,
            'Winner': row['Winner'],
            'Loser': row['Loser'],
            'gain_net': (oddb - 1) if B == row['Winner'] else -1,
            'pari_sur': B
        }

        # X = pd.DataFrame([{k: feat_A[k] for k in features}])
        # X = X.apply(pd.to_numeric, errors='coerce')

        # probaLGBM = modelLGBM.predict_proba(X)[0][1]

        # if probaLGBM > 0.5:
        #     records.append(feat_A)
        # else:
        #     records.append(feat_B)

        coteBookmaker = 1/odda
        if coteBookmaker > 0.5:
            records.append(feat_A)
        else:
            records.append(feat_B)

        # Mise à jour ELO
        result_A = target
        result_B = 1 - target

        expected_A = 1 / (1 + 10 ** ((statsB['elo'] - statsA['elo']) / 400))
        new_elo_A = statsA['elo'] + k_elo * (result_A - expected_A)
        new_elo_B = statsB['elo'] + k_elo * (result_B - (1 - expected_A))

        expected_surf_A = 1 / (1 + 10 ** ((statsB[f'elo_{surface}'] - statsA[f'elo_{surface}']) / 400))
        new_elo_surf_A = statsA[f'elo_{surface}'] + k_elo * (result_A - expected_surf_A)
        new_elo_surf_B = statsB[f'elo_{surface}'] + k_elo * (result_B - (1 - expected_surf_A))

        # Mises à jour des historiques
        wins[A] += result_A
        wins[B] += result_B
        games[A] += 1
        games[B] += 1
        surface_win[(A, surface)] += result_A
        surface_win[(B, surface)] += result_B
        surface_games[(A, surface)] += 1
        surface_games[(B, surface)] += 1

        last50 [A].append(result_A)
        last50 [B].append(result_B)
        last100[A].append(result_A)
        last100[B].append(result_B)
        recent_games_diff[A].append(1 if result_A else -1)
        recent_games_diff[B].append(1 if result_B else -1)
        recent_decider[A].append(1 if result_A else 0)
        recent_decider[B].append(1 if result_B else 0)

        # Update H2H
        if A == row['Winner']:
            h2h[(A, B)] += 1
            h2h_surface[(A, B, surface)] += 1
        else:
            h2h[(B, A)] += 1
            h2h_surface[(B, A, surface)] += 1

        # Mise à jour des stats finales
        player_stats_dict[A] = {
            **statsA,
            'elo': new_elo_A,
            f'elo_{surface}': new_elo_surf_A,
            f'mean_{surface}': surface_win[(A, surface)] / surface_games[(A, surface)],
            'wrlast50': np.mean(last50[A]),
            'wrlast100': np.mean(last100[A]),
            'winrate': wins[A] / games[A],
            'rank': rank_a,
            'pts': pts_a,
            'recent_games_diff_last': recent_games_diff[A][-1],
            'recent_decider_last': recent_decider[A][-1]
        }

        player_stats_dict[B] = {
            **statsB,
            'elo': new_elo_B,
            f'elo_{surface}': new_elo_surf_B,
            f'mean_{surface}': surface_win[(B, surface)] / surface_games[(B, surface)],
            'wrlast50': np.mean(last50[B]),
            'wrlast100': np.mean(last100[B]),
            'winrate': wins[B] / games[B],
            'rank': rank_b,
            'pts': pts_b,
            'recent_games_diff_last': recent_games_diff[B][-1],
            'recent_decider_last': recent_decider[B][-1]
        }

    dataset = pd.DataFrame(records)
    return dataset, player_stats_dict, h2h, h2h_surface



print("Building match dataset...")
df = df.sort_values(by="Date")
match_df, player_stats_final, h2h, h2h_surface = build_match_dataset_online(df)

# 4

from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report

# Entraînement initial pour évaluer sur 2025
# def train_model(X, y):
#     model = LGBMClassifier(
#         n_estimators=1000,
#         learning_rate=0.02,
#         max_depth=6,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         min_child_samples=50,
#         min_split_gain=0.01,
#         reg_alpha=1.0,
#         reg_lambda=1.0,
#         random_state=42
#     )

#     model.fit(X, y)
#     return model

# def evaluate_model(model, X_test, y_test):
#     y_pred = model.predict(X_test)
#     acc = accuracy_score(y_test, y_pred)
#     print(f"✅ Accuracy sur données de test (2025): {acc:.4f}")
#     print("📊 Rapport de classification :")
#     print(classification_report(y_test, y_pred))
#     return acc

# # Réentraînement sur toutes les données disponibles (2015–2025)
# def retrain_final_model(match_train, match_test):
#     match_total = pd.concat([match_train, match_test], ignore_index=True)
#     X_total = match_total.drop(columns=['Winner', 'Loser', 'target', 'Date', 'Surface', 'PlayerA', 'PlayerB', 'OddA', 'OddB'])
#     y_total = match_total['target']
#     print("🔁 Réentraînement du modèle final sur l’ensemble des données (2015–2025)...")
#     final_model = train_model(X_total, y_total)
#     return final_model

# print("Training model with LGBM...")

# match_train = match_df[match_df['Date'] < '2025-01-01']
# match_test = match_df[match_df['Date'] >= '2025-01-01']

# X_train = match_train.drop(columns=['Winner', 'Loser', 'target', 'Date', 'Surface', 'PlayerA', 'PlayerB', 'OddA', 'OddB'])
# y_train = match_train['target']


# X_test = match_test.drop(columns=['Winner', 'Loser', 'target', 'Date', 'Surface', 'PlayerA', 'PlayerB', 'OddA', 'OddB'])
# y_test = match_test['target']

# model = train_model(X_train, y_train)

# final_model = retrain_final_model(match_train, match_test)
# evaluate_model(model, X_test, y_test)

from lightgbm import LGBMRegressor

features = [
    'elo_diff', 'elo_surface_diff', 'winrate_diff', 'surface_winrate_diff',
    'wrlast50_diff', 'wrlast100_diff', 'h2h_diff', 'h2h_surface_diff', 'odds-ratio',
    'level', 'round', 'games_diff_recent', 'decider_winrate_diff'
]

match_df_train = match_df[match_df['Date'] < '2025-01-01']
match_df_test = match_df[match_df['Date'] >= '2025-01-01']

X_train = match_df_train[features]
y_train = match_df_train['gain_net']

X_test = match_df_test[features]
y_test = match_df_test['gain_net']

model_gain = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)
model_gain.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_true = y_test  # tes vraies valeurs de gain_net
y_pred = model_gain.predict(X_test)  # les prédictions

mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
y_true_clean = y_true[mask]
y_pred_clean = y_pred[mask]

print("MAE :", mean_absolute_error(y_true, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
print("R²  :", r2_score(y_true, y_pred))

print(match_df['gain_net'].describe())
print("Nombre de gains positifs :", (match_df['gain_net'] > 0).sum())
print("Nombre de gains nuls ou négatifs :", (match_df['gain_net'] <= 0).sum())

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(match_df_test['gain_net'], bins=50, edgecolor='black', color='skyblue')
plt.title('Distribution des gains réels (gain_net)')
plt.xlabel('Gain net')
plt.ylabel('Nombre de matchs')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y_true, y_pred, alpha=0.4)
plt.plot([-1, 5], [-1, 5], color='red', linestyle='--')  # diagonale parfaite
plt.title('Gain réel vs Gain prédit')
plt.xlabel('Gain réel')
plt.ylabel('Gain prédit')
plt.grid(True)
plt.show()


def evaluate_bets_by_gain_model(test_df, model_gain, features, seuil_gain=0.05, bankroll=100):
    bankroll_init = bankroll
    gains = []

    for _, row in test_df.iterrows():
        row_A = {f: row[f] for f in features}
        row_B = {f: -row[f] if 'diff' in f else row[f] for f in features}

        X_A = pd.DataFrame([row_A])
        X_B = pd.DataFrame([row_B])

        gain_pred_A = model_gain.predict(X_A)[0]
        gain_pred_B = model_gain.predict(X_B)[0]

        if gain_pred_A > gain_pred_B:
            joueur = row['PlayerA']
            gain_pred = gain_pred_A
            odd = row['OddA']
        else:
            joueur = row['PlayerB']
            gain_pred = gain_pred_B
            odd = row['OddB']

        if gain_pred < seuil_gain:
            continue

        mise = 1  # ou kelly stake
        gain = (odd - 1) * mise if row['Winner'] == joueur else -mise
        bankroll += gain
        gains.append(gain)

    print(f"Bilan : {bankroll:.2f}€ | ROI : {(bankroll - bankroll_init)/bankroll_init:.2%}")
    return gains

evaluate_bets_by_gain_model(match_df_test, model_gain, features)


# Calibration

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

# model = CalibratedClassifierCV(model, method='sigmoid', cv=3)
# model.fit(X_train, y_train)

# from sklearn.metrics import accuracy_score, classification_report

# def calibrate_for_profit(p_raw, y_true, odds,
#                          bet_threshold=0.02,
#                          kelly_fraction=1.0):  # fraction de Kelly (ex: 0.5 = prudence)

#     eps = 1e-6
#     p_raw = np.clip(p_raw, eps, 1 - eps)
#     z = logit(p_raw)

#     def objective(params):
#         a, b = params
#         p_cal = expit(a * z + b)

#         b_odds = odds - 1
#         q_cal = 1 - p_cal

#         # Kelly stake (fraction de bankroll)
#         kelly_raw = (b_odds * p_cal - q_cal) / b_odds
#         kelly_stake = np.maximum(kelly_raw, 0) * kelly_fraction  # pas de mises négatives

#         # EV “ex‑ante”
#         ev = p_cal * b_odds - q_cal

#         # Filtrage selon seuil d'EV
#         mask = ev > bet_threshold
#         if not np.any(mask):
#             return 0.0

#         # Profit pondéré par la mise Kelly
#         gain = y_true[mask] * b_odds[mask] - (1 - y_true[mask])
#         profit = gain * kelly_stake[mask]

#         return -np.sum(profit)

#     res = minimize(objective, x0=[1.0, 0.0], method="Nelder-Mead")
#     return res.x

# ---- split train / val -------------------------------------------------
# X_train_final, X_val, y_train_final, y_val = train_test_split(
#     X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
# )

# # ---- entraînement “brut” ----------------------------------------------
# model_raw = train_model(X_train_final, y_train_final)

# # ---- proba brutes sur la validation -----------------------------------
# p_val_raw = model_raw.predict_proba(X_val)[:, 1]

# # ---- récupère la cote du joueur A -------------------------------------
# odds_val = match_train['OddA'].values  # ← colonne déjà dans ton dataset

# ---- calibration profit‑aware -----------------------------------------
# a_opt, b_opt = calibrate_for_profit(p_val_raw, y_val.values, odds_val)

# print(f"📈 Paramètres profit-aware : a = {a_opt:.3f} | b = {b_opt:.3f}")

# print(X_train.var())
# print(match_df['target'].value_counts(normalize=True))
# import numpy as np
# import matplotlib.pyplot as plt

# # Comptage des victoires réelles par joueur
# victoires_reelles = match_df['target'].value_counts()

# print(victoires_reelles)

# joblib.dump(model, "modeleLGBM_tennis_2024-" + str(seed) + ".pkl")
# joblib.dump(player_stats_final, "players_stats.pkl")
# joblib.dump(final_model, "modeleLGBM_tennis_2025-" + str(seed) + ".pkl")
# joblib.dump(match_df, "match_df.pkl")
# joblib.dump(h2h, "h2h.pkl")
# joblib.dump(h2h_surface, "h2h_surface.pkl")
# joblib.dump(df, "df_tennis.pkl")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importances(model, feature_names, top_n=20):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = model.coef_[0]
    else:
        raise ValueError("Le modèle ne fournit pas d'importances de features directement.")

    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(top_n), palette="viridis")
    plt.title(f"📊 Top {top_n} Features les plus utilisées")
    plt.tight_layout()
    plt.show()

    return feature_importance_df

plot_feature_importances(model, X_train.columns)

from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(y_test, model.predict_proba(X_test)[:, 1], n_bins=10)
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("Courbe de calibration")
plt.xlabel("Proba prédite")
plt.ylabel("Proba réelle")
plt.show()