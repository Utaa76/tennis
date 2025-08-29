import numpy as np
import pandas as pd
import joblib

modelLGBM = joblib.load("modeleLGBM_tennis_2024-20.pkl")
iso = joblib.load("calibrateur-20.pkl")
modelLGBMfinal = joblib.load("modeleLGBM_tennis_2025-20.pkl")
player_stats = joblib.load("players_stats.pkl")
match_df = joblib.load("match_df.pkl")
match_df_test = match_df[match_df['Date'] >= '2024-01-01']
h2h = joblib.load("h2h.pkl")
h2h_surface = joblib.load("h2h_surface.pkl")

# Maps “tour level” & “round”
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

from scipy.special import logit, expit
import numpy as np

def apply_profit_calibration(p_raw, a, b):
    """Applique p_cal = σ(a·logit(p_raw) + b) élément‑par‑élément."""
    eps = 1e-6
    p_raw = np.clip(p_raw, eps, 1 - eps)        # évite inf/‑inf
    return expit(a * logit(p_raw) + b)

def default_player_stats(player_name):
    return {
        'Player': player_name,
        'matches_played': 0,
        'wins': 0,
        'winrate': 0.5,
        'elo': 1500,
        'elo_hard': 1500,
        'elo_clay': 1500,
        'elo_grass': 1500,
        'elo_carpet': 1500,
        'recent_form': 0.5,
        'mean_clay': 0.5,
        'mean_grass': 0.5,
        'mean_hard': 0.5,
        'recent_games_diff_last': 0,
        'recent_decider_last': 0.5,
        'rank': 500,
        'pts': 0,
        'wrlast50': 0.25,
        'wrlast100': 0.25
    }

def get_player_stats(player_name, player_stats_df):
    if player_name in player_stats:
        return player_stats_df[player_name]
    else:
        return default_player_stats(player_name)

import matplotlib.pyplot as plt

def plot_elo_evolution(dataset, player_name):
    # Filtrer les matchs où le joueur a joué (en tant que Winner ou Loser)
    player_matches = dataset[(dataset['Winner'] == player_name) | (dataset['Loser'] == player_name)].copy()
    player_matches = player_matches.sort_values('Date')

    # Initialiser liste pour stocker l'Elo du joueur à chaque match
    elos = []
    dates = []

    # Elo initial à 1500 (valeur par défaut)
    current_elo = 1500

    for _, row in player_matches.iterrows():
        # On récupère le différentiel Elo avant le match (elo_diff + elo_surface_diff, etc)
        # Ici on va calculer directement l'Elo à partir des infos dans le dataset :
        # On peut approximer : elo_joueur = elo_opp + elo_diff si joueur = Winner
        # Mais c’est plus simple de reconstruire en réappliquant les updates
        # Comme on n’a pas l’Elo absolu, on peut faire ça par cumul :
        # Ou bien, comme la fonction build_match_dataset_online ne conserve pas les Elo à chaque match,
        # Il faudrait modifier la fonction pour stocker l’Elo après chaque match.
        # Sinon, on peut approximer en utilisant elo_diff.

        # Pour simplifier, on peut stocker directement l’elo_diff à chaque match
        # Si joueur est A (Winner ou Loser)
        if row['Winner'] == player_name:
            elo = 1500 + row['elo_diff']  # approx (attention, ce n'est qu'une estimation)
        else:
            elo = 1500 - row['elo_diff']

        elos.append(elo)
        dates.append(row['Date'])

    # Tracer
    plt.figure(figsize=(10,6))
    plt.plot(dates, elos, marker='o')
    plt.title(f"Evolution de l'Elo estimé de {player_name}")
    plt.xlabel("Date")
    plt.ylabel("Elo (approximation)")
    plt.grid(True)
    plt.show()

def build_upcoming_dataset(upcoming_matches_df, player_stats):
    records = []

    for _, row in upcoming_matches_df.iterrows():
        player_A = row['Player1']
        player_B = row['Player2']
        surface = row['Surface']
        odds_A = row.get('Odds_A', 1.8)
        odds_B = row.get('Odds_B', 2.0)
        surface_key = f'elo_{surface.lower()}'
        mean_surface_key = f'mean_{surface}'
        level = row['Series']
        round = row.get('Round', '2nd Round')

        predict_match(player_A, player_B, surface, odds_A, odds_B, level, round)

        # Récupération des stats des joueurs
        stats_A = get_player_stats(player_A, player_stats)
        stats_B = get_player_stats(player_B, player_stats)

        if stats_A.empty or stats_B.empty:
            print("No stats")
            continue

        # Convertir en dictionnaire simple si besoin
        row_A = stats_A.iloc[0]
        row_B = stats_B.iloc[0]

        # Head-to-head diff
        past_matches = df[
            ((df['Winner'] == player_A) & (df['Loser'] == player_B)) |
            ((df['Winner'] == player_B) & (df['Loser'] == player_A))
        ]
        wins_A = past_matches[past_matches['Winner'] == player_A].shape[0]
        wins_B = past_matches[past_matches['Winner'] == player_B].shape[0]
        total = wins_A + wins_B if wins_A + wins_B > 0 else 1
        h2h_diff = (wins_A / total) - (wins_B / total)

        # Log odds ratio
        log_odds_ratio = np.log(odds_B / odds_A) if odds_A > 0 and odds_B > 0 else 0

        # Création des features
        features = {
            'elo_diff': float(row_A['elo']) - float(row_B['elo']),
            'elo_surface_diff': float(row_A.get(surface_key, 1500)) - float(row_B.get(surface_key, 1500)),
            'winrate_diff': float(row_A['winrate']) - float(row_B['winrate']),
            'surface_winrate_diff': float(row_A.get(mean_surface_key, 0)) - float(row_B.get(mean_surface_key, 0)),
            'recent_form_diff': float(row_A['recent_form']) - float(row_B['recent_form']),
            'h2h-diff': h2h_diff,
            'odds-ratio': log_odds_ratio,
            'elo_surface': (float(row_A['elo']) - float(row_B['elo'])) *
                           (float(row_A.get(surface_key, 1500)) - float(row_B.get(surface_key, 1500)))
        }

        records.append({
            'Player1': player_A,
            'Player2': player_B,
            'Surface': surface,
            **features
        })

    return pd.DataFrame(records)

def predict_proba_calibrated(model, X):
    raw = model.predict(X)
    clipped = np.clip(raw, 0, 1)
    calibrated = iso.predict(clipped)
    return calibrated

def remove_margin(odds_list):
    implied_probs = [1/o for o in odds_list]
    total = sum(implied_probs)
    fair_probs = [p/total for p in implied_probs]
    fair_odds = [1/p for p in fair_probs]
    return fair_odds

def apply_margin(odds_list):
    fair_probs = [1/o for o in odds_list]
    total = sum(fair_probs)
    unfair_probs = [p/total for p in fair_probs]
    unfair_odds = [0.95/p for p in unfair_probs]
    return unfair_odds

# 5
def predict_match(
    A, B, surface,
    cote_A, cote_B,
    level_name,                # ex. 'Grand Slam', 'ATP500', …
    round_name,                # ex. 'Quarterfinals', 'Semifinals', …
    rmv_margin=False,
    ev_comparison=False,
    bankroll=100, min_ev=0.05  # mettre 0.1 ?
):
    """
    Renvoie la décision de pari pour un seul match A‑B à `surface`
    avec les cotes Bet365 (ou autres) déjà connues.
    """

    # ---------- mapping tour & round (même qu’off‑line) ----------
    tournament_level_map = {
        "Grand Slam": 4,
        "Masters 1000": 3,
        "ATP500": 2,
        "ATP250": 1,
        "WTA250": 1,
        "WTA500": 2,
        "WTA1000": 3
    }
    round_map = {
        '1st Round': 1, '2nd Round': 2, '3rd Round': 3, '4th Round': 4,
        'Quarterfinals': 5, 'Semifinals': 6, 'The Final': 7
    }

    # ---------- récupérer les stats joueurs ----------
    statsA = get_player_stats(A, player_stats)
    statsB = get_player_stats(B, player_stats)

    # if statsA['elo'] < statsB['elo']:
    #     statsA, statsB = statsB, statsA
    #     A, B = B, A
    #     cote_A, cote_B = cote_B, cote_A

    # ---------- vérifs ----------
    if A not in player_stats:
        print(f"⚠️ Joueur {A} manquant dans les stats. Skip.\n")
        return None
    if B not in player_stats:
        print(f"⚠️ Joueur {B} manquant dans les stats. Skip.\n")
        return None

    df_stats = pd.DataFrame({
        f'{A}': statsA,
        f'{B}': statsB
    })

    # print(df_stats)

    if statsB[f'elo_{surface.lower()}'] > statsA[f'elo_{surface.lower()}']:
        A, B = B, A
        statsA, statsB = statsB, statsA
        cote_A, cote_B = cote_B, cote_A

    # ---------- H2H ----------
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

    lst = remove_margin([cote_A, cote_B])
    
    oddA, oddB = lst[0], lst[1]
    bookOddA, bookOddB = cote_A, cote_B

    # ---------- features ----------
    log_odds_ratio = np.log(oddB / oddA) * 5 if oddA > 0 and oddB > 0 else 0
    feat = pd.DataFrame([{
        'elo_diff': statsA['elo'] - statsB['elo'],
        'elo_surface_diff': statsA.get(f'elo_{surface.lower()}', 1500) -
                            statsB.get(f'elo_{surface.lower()}', 1500),
        'winrate_diff': statsA['winrate'] - statsB['winrate'],
        'surface_winrate_diff': statsA.get(f'mean_{surface.lower()}', 0) -
                                statsB.get(f'mean_{surface.lower()}', 0),
        'wrlast50_diff': statsA['wrlast50'] - statsB['wrlast50'],
        'wrlast100_diff': statsA['wrlast100'] - statsB['wrlast100'],
        'h2h_diff': h2h_diff,
        'h2h_surface_diff': h2h_surface_diff,
        'odds-ratio': log_odds_ratio,
        'level': tournament_level_map.get(level_name, 1),
        'round': round_map.get(round_name, 1),
        'rank_diff': statsA['rank'] - statsB['rank'],
        'pts_diff': statsA['pts'] - statsB['pts'],
    }]).astype(float)

    if not rmv_margin:
        oddA, oddB = cote_A, cote_B

    # ---------- probabilités modèles ----------
    # probaLGBM = modelLGBMfinal.predict_proba(feat)[0, 1] # mettre modelLGBMfinal !!!
    probaLGBM = predict_proba_calibrated(modelLGBM, feat)[0]

    probA = probaLGBM
    probB = 1 - probA

    # EV des deux côtés
    evA = (probA * (oddA - 1)) - (1 - probA)
    evB = (probB * (oddB - 1)) - (1 - probB)

    # Kelly des deux côtés
    kellyA = (probA * (oddA - 1) - (1 - probA)) / (oddA - 1)
    kellyB = (probB * (oddB - 1) - (1 - probB)) / (oddB - 1)

    # Sélection via Kelly (même si EV < 0, tu choisis le + gros Kelly)
    if kellyA > kellyB:
        winner, odd, cote, p_win, kelly = A, oddA, bookOddA, probA, kellyA
    else:
        winner, odd, cote, p_win, kelly = B, oddB, bookOddB, probB, kellyB

    print(f"🎾 \033[1m{A} vs {B}\033[0m ({surface}) — pari : \033[4m{winner}\033[0m 🏆")
    print(f"\t📊 Probabilité modèle : {p_win:.2%} | Cote : {cote}")

    # Vérification EV après sélection
    ev = (p_win * (odd - 1)) - (1 - p_win)
    if ev < min_ev:
        print(f"\t⚠️ \033[91m EV {ev:.4f} < {min_ev}, aucun pari. 🚫\033[0m")
        print("\n")
        return None

    # ---------- Kelly ----------
    mise = kelly * bankroll * 0.25
    if mise < 0.1:
        print(f"\t⚠️ \033[91m Mise ({mise:.2f} €) trop faible, aucun pari. 💸\033[0m")
        print("\n")
        return None

    gain_attendu = (cote - 1) * mise

    # ---------- affichage ----------
    print(f"\t\t💰 EV : {ev:.4f} | Mise (Kelly) : \033[1;32m{mise:.2f} €\033[0m")
    print(f"\t\t🎯 Gain net attendu : \033[1;32m{gain_attendu:.2f} €\033[0m")
    print("\n")

    return {
        'match': f"{A} vs {B}",
        'joueur1': A,
        'joueur2': B,
        'winner': winner,
        'surface': surface,
        'probability': p_win,
        'cote': cote,
        'expected_value': ev,
        'mise': mise,
        'gain_attendu': gain_attendu,
        'round': round_name,
        'level': level_name
    }



def evaluate_bets_online(dataset, bankroll=100, min_ev=0.1):
    results = []
    combines = []

    correct = 0

    for idx, row in dataset.iterrows():
        if idx % 100 == 0:
            print(f"Traitement match #{idx}")

        playerA = row['PlayerA']
        playerB = row['PlayerB']
        surface = row['Surface'] if 'Surface' in row else None
        date = row['Date'] if 'Date' in row else None
        coteA = row['OddA']
        coteB = row['OddB']

        lst = remove_margin([coteA, coteB])
        oddA, oddB = lst[0], lst[1]
        unfairOdds = apply_margin([oddA, oddB])
        unfairOddA, unfairOddB = unfairOdds[0], unfairOdds[1]

        if np.isnan(coteA) or np.isnan(coteB):
            continue

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
        ]

        # print(playerA, "vs", playerB, "Surface :", surface, "Cotes : ", coteA, " ; ", coteB)
        # print(row[features])

        row['odds-ratio'] = np.log(oddB / oddA) * 5 if oddA > 0 and oddB > 0 else 0

        X = row[features].to_frame().T
        X = X.apply(pd.to_numeric, errors='coerce')

        # Récupère les scores bruts (logits)
        # p_raw = modelLGBM.predict(X)[0]
        # p_cal = p_raw
        probA = predict_proba_calibrated(modelLGBM, X)[0]
        probB = 1 - probA

        evA = (probA * (unfairOddA - 1)) - (1 - probA)
        evB = (probB * (unfairOddB - 1)) - (1 - probB)

        if (max(evA, evB) <= 0):
            continue

        if (evA > evB):
            proba_effective = probA
            player_pred = playerA
            cote_pred = unfairOddA
            ev = evA
        else:
            proba_effective = probB
            player_pred = playerB
            cote_pred = unfairOddB
            ev = evB

        if (player_pred == playerA and get_player_stats(playerB, player_stats)['rank'] <= 10) or (player_pred == playerB and get_player_stats(playerA, player_stats)['rank'] <= 10):
            continue

        if row['Winner'] == player_pred:
            correct += 1

        if ev < min_ev:
            if ev > 0:
                combines.append({
                    'Date': date,
                    'Player A': row['Winner'],
                    'Player B': row['Loser'],
                    'Joueur predit': player_pred,
                    'Surface': surface,
                    'Proba': proba_effective,
                    'Cote_pred': cote_pred,
                    'EV': ev,
                    'Resultat': 1 if row['Winner'] == player_pred else 0
                })
            continue
        
        b = cote_pred - 1
        p = proba_effective
        q = 1 - p
        fraction_kelly = max((b * p - q) / b, 0)
        mise = fraction_kelly * bankroll * 0.25

        if mise < 0.1:
            continue

        gain_net = (cote_pred - 1) * mise if row['Winner'] == player_pred else -mise
        bankroll += gain_net
        # print("New bankroll : ", bankroll)

        results.append({
            'Date': date,
            'Player A': row['Winner'],
            'Player B': row['Loser'],
            'Joueur predit': player_pred,
            'Surface': surface,
            'Proba': proba_effective,
            'Cote_pred': cote_pred,
            'EV': ev,
            'Mise': mise,
            'Gain net': gain_net,
            'Resultat': 1 if row['Winner'] == player_pred else 0
        })



    print("Précision : ", correct / len(dataset))

    return pd.DataFrame(results), pd.DataFrame(combines)



def simulation_paris_2025():
    # Exemple d’utilisation sur df_test (matches 2025+)
    print("Evaluation des paris sur les matchs de test...")
    df_bets_results, df_combines = evaluate_bets_online(match_df_test)
    print(df_bets_results.sort_values(by='Gain net'))
    print("💸 Nombre de paris passés :", len(df_bets_results), "sur", len(match_df_test), "matchs")
    print("    Total argent mis en jeu :", df_bets_results['Mise'].sum())
    print("📈 Gain total :", df_bets_results['Gain net'].sum(), "€")
    print("📊 Yield :", df_bets_results['Gain net'].sum() / (df_bets_results['Mise'].sum()))

    import matplotlib.pyplot as plt

    # Calcul de l'évolution de la bankroll
    df_bets_results['Cumulative Gain'] = df_bets_results['Gain net'].cumsum()
    df_bets_results['Bankroll'] = 100 + df_bets_results['Cumulative Gain']  # 100 = bankroll initiale

    # Tracé de la courbe de bankroll
    plt.figure(figsize=(12, 6))
    plt.plot(df_bets_results['Bankroll'], label='Bankroll (€)', color='green')
    plt.axhline(y=100, color='gray', linestyle='--', label='Bankroll initiale')
    plt.title("📈 Évolution de la bankroll")
    plt.xlabel("Nombre de paris")
    plt.ylabel("Bankroll (€)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    df_bets_results['Gain net'].hist(bins=30)
    plt.title("Distribution des gains nets")
    plt.show()

    df_combines = df_combines.sort_values(by="Date")
    pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    df_combines = df_combines[df_combines['Date'] >= '2025-01-01']
    print(df_combines)

    losing_bets = df_bets_results[(df_bets_results['Resultat'] == 0) & (df_bets_results['EV'] > 0)]

    print(losing_bets['Joueur predit'].value_counts().head(10))  # Top 10 joueurs sur lesquels tu perds
    print(losing_bets['Surface'].value_counts())                # Surfaces les plus "risquées"
    print(losing_bets['EV'].describe())                         # Distribution de l’EV perdant
    print(losing_bets['Cote_pred'].describe())                  # Distribution des cotes sur paris perdants

    # Moyenne EV perdue par joueur
    mean_ev_per_player = losing_bets.groupby('Joueur predit')['EV'].mean().sort_values()

    # Comptage des pertes par surface
    losses_by_surface = losing_bets.groupby('Surface').size()

    print(mean_ev_per_player.head(10))
    print(losses_by_surface)

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.hist(losing_bets['EV'], bins=20)
    plt.title('Distribution EV des paris perdants')
    plt.xlabel('EV')
    plt.ylabel('Nombre de paris')
    plt.show()

    # Heatmap joueur x surface (nombre de paris perdants)
    pivot = losing_bets.pivot_table(index='Joueur predit', columns='Surface', aggfunc='size', fill_value=0)
    sns.heatmap(pivot, cmap='Reds')
    plt.title('Paris perdants par joueur et surface')
    plt.show()

    print(losing_bets.sort_values('EV', ascending=False).head(10))

    sns.boxplot(x='Resultat', y='elo_diff', data=df_bets_results) # crash ici car elo_diff n'existe pas dans df_bets_results
    plt.title('Elo diff: gagnants vs perdants')
    plt.show()

    losing_bets['Date'] = pd.to_datetime(losing_bets['Date'])
    losing_bets.groupby(losing_bets['Date'].dt.month).size().plot(kind='bar')
    plt.title('Nombre de paris perdants par mois')
    plt.show()




    # Tu peux ensuite analyser df_bets_results, par ex. :
    # df_bets_results[df_bets_results['EV'] > 0].describe()


import seaborn as sns
import matplotlib.pyplot as plt

# Liste construite à partir du tirage officiel PDF de Wimbledon 2025
# Forme : "Player1" vs "Player2"
data = [
  {
    "Player1": "Tarvet O.",
    "Player2": "Riedi L.",
    "Odds_A": 2.1,
    "Odds_B": 1.75,
    "Round": "1st Round"
  },
  {
    "Player1": "Lehecka J.",
    "Player2": "Dellien H.",
    "Odds_A": 1.02,
    "Odds_B": 15.0,
    "Round": "1st Round"
  },
  {
    "Player1": "Thompson J.",
    "Player2": "Kopriva V.",
    "Odds_A": 1.4,
    "Odds_B": 3.0,
    "Round": "1st Round"
  },
  {
    "Player1": "O Connell C.",
    "Player2": "Mannarino A.",
    "Odds_A": 2.9,
    "Odds_B": 1.36,
    "Round": "1st Round"
  },
  {
    "Player1": "Bellucci M.",
    "Player2": "Crawford O.",
    "Odds_A": 1.3,
    "Odds_B": 3.6,
    "Round": "1st Round"
  },
  {
    "Player1": "Bonzi B.",
    "Player2": "Medvedev D.",
    "Odds_A": 8.5,
    "Odds_B": 1.07,
    "Round": "1st Round"
  },
  {
    "Player1": "Basavareddy N.",
    "Player2": "Tien L.",
    "Odds_A": 2.6,
    "Odds_B": 1.5,
    "Round": "1st Round"
  },
  {
    "Player1": "Tiafoe F.",
    "Player2": "Moller E.",
    "Odds_A": 1.04,
    "Odds_B": 9.0,
    "Round": "1st Round"
  },
  {
    "Player1": "Norrie C.",
    "Player2": "Bautista Agut R.",
    "Odds_A": 1.83,
    "Odds_B": 1.91,
    "Round": "1st Round"
  },
  {
    "Player1": "Fery A.",
    "Player2": "Popyrin A.",
    "Odds_A": 4.33,
    "Odds_B": 1.22,
    "Round": "1st Round"
  },
  {
    "Player1": "Bergs Z.",
    "Player2": "Harris L.",
    "Odds_A": 1.5,
    "Odds_B": 2.5,
    "Round": "1st Round"
  },
  {
    "Player1": "Jarry N.",
    "Player2": "Rune H.",
    "Odds_A": 3.75,
    "Odds_B": 1.29,
    "Round": "1st Round"
  },
  {
    "Player1": "Tsitsipas S.",
    "Player2": "Royer V.",
    "Odds_A": 1.25,
    "Odds_B": 4.0,
    "Round": "1st Round"
  },
  {
    "Player1": "Fognini F.",
    "Player2": "Alcaraz C.",
    "Odds_A": 17.0,
    "Odds_B": 1.01,
    "Round": "1st Round"
  },
  {
    "Player1": "Darderi L.",
    "Player2": "Safiullin R.",
    "Odds_A": 3.0,
    "Odds_B": 1.4,
    "Round": "1st Round"
  },
  {
    "Player1": "Cerundolo F.",
    "Player2": "Borges N.",
    "Odds_A": 1.53,
    "Odds_B": 2.4,
    "Round": "1st Round"
  },
  {
    "Player1": "Duckworth J.",
    "Player2": "Auger-Aliassime F.",
    "Odds_A": 4.5,
    "Odds_B": 1.18,
    "Round": "1st Round"
  },
  {
    "Player1": "Fonseca J.",
    "Player2": "Fearnley J.",
    "Odds_A": 1.5,
    "Odds_B": 2.63,
    "Round": "1st Round"
  },
  {
    "Player1": "Quinn E.",
    "Player2": "Searle H.",
    "Odds_A": 1.44,
    "Odds_B": 2.8,
    "Round": "1st Round"
  },
  {
    "Player1": "Harris B.",
    "Player2": "Lajovic D.",
    "Odds_A": 1.4,
    "Odds_B": 2.8,
    "Round": "1st Round"
  },
  {
    "Player1": "Khachanov K.",
    "Player2": "McDonald M.",
    "Odds_A": 1.22,
    "Odds_B": 4.33,
    "Round": "1st Round"
  },
  {
    "Player1": "Garin C.",
    "Player2": "Rodesch C.",
    "Odds_A": 1.67,
    "Odds_B": 2.2,
    "Round": "1st Round"
  },
  {
    "Player1": "Djere L.",
    "Player2": "Rublev A.",
    "Odds_A": 4.0,
    "Odds_B": 1.22,
    "Round": "1st Round"
  },
  {
    "Player1": "Misolic F.",
    "Player2": "Struff J.L.",
    "Odds_A": 2.25,
    "Odds_B": 1.67,
    "Round": "1st Round"
  },
  {
    "Player1": "Berrettini M.",
    "Player2": "Majchrzak K.",
    "Odds_A": 1.13,
    "Odds_B": 6.0,
    "Round": "1st Round"
  },
  {
    "Player1": "Holt B.",
    "Player2": "Davidovich Fokina A.",
    "Odds_A": 4.33,
    "Odds_B": 1.22,
    "Round": "1st Round"
  },
  {
    "Player1": "Griekspoor T.",
    "Player2": "Brooksby J.",
    "Odds_A": 1.4,
    "Odds_B": 2.75,
    "Round": "1st Round"
  },
  {
    "Player1": "Diallo G.",
    "Player2": "Altmaier D.",
    "Odds_A": 1.2,
    "Odds_B": 4.5,
    "Round": "1st Round"
  },
  {
    "Player1": "Tseng C.H.",
    "Player2": "Vukic A.",
    "Odds_A": 4.0,
    "Odds_B": 1.25,
    "Round": "1st Round"
  },
  {
    "Player1": "McCabe J.",
    "Player2": "Marozsan F.",
    "Odds_A": 2.8,
    "Odds_B": 1.44,
    "Round": "1st Round"
  },
  {
    "Player1": "Kecmanovic M.",
    "Player2": "Michelsen A.",
    "Odds_A": 2.8,
    "Odds_B": 1.44,
    "Round": "1st Round"
  },
  {
    "Player1": "Faria J.",
    "Player2": "Sonego L.",
    "Odds_A": 3.6,
    "Odds_B": 1.3,
    "Round": "1st Round"
  },
  {
    "Player1": "Monday J.",
    "Player2": "Paul T.",
    "Odds_A": 8.5,
    "Odds_B": 1.05,
    "Round": "1st Round"
  },
  {
    "Player1": "De Minaur A.",
    "Player2": "Carballes Baena R.",
    "Odds_A": 1.01,
    "Odds_B": 13.0,
    "Round": "1st Round"
  },
  {
    "Player1": "Walton A.",
    "Player2": "Cazaux A.",
    "Odds_A": 2.5,
    "Odds_B": 1.53,
    "Round": "1st Round"
  },
  {
    "Player1": "Basilashvili N.",
    "Player2": "Musetti L.",
    "Odds_A": 3.4,
    "Odds_B": 1.3,
    "Round": "1st Round"
  },
  {
    "Player1": "De Jong J.",
    "Player2": "Eubanks C.",
    "Odds_A": 1.75,
    "Odds_B": 2.1,
    "Round": "1st Round"
  },
  {
    "Player1": "Navone M.",
    "Player2": "Shapovalov D.",
    "Odds_A": 6.0,
    "Odds_B": 1.13,
    "Round": "1st Round"
  },
  {
    "Player1": "Bublik A.",
    "Player2": "Munar J.",
    "Odds_A": 1.22,
    "Odds_B": 4.33,
    "Round": "1st Round"
  },
  {
    "Player1": "Sinner J.",
    "Player2": "Nardi L.",
    "Odds_A": 1.01,
    "Odds_B": 13.0,
    "Round": "1st Round"
  },
  {
    "Player1": "Medjedovic H.",
    "Player2": "Ofner S.",
    "Odds_A": 1.53,
    "Odds_B": 2.5,
    "Round": "1st Round"
  },
  {
    "Player1": "Loffhagen G.",
    "Player2": "Martinez P.",
    "Odds_A": 1.3,
    "Odds_B": 3.6,
    "Round": "1st Round"
  },
  {
    "Player1": "Van De Zandschulp B.",
    "Player2": "Arnaldi M.",
    "Odds_A": 2.25,
    "Odds_B": 1.67,
    "Round": "1st Round"
  },
  {
    "Player1": "Zeppieri G.",
    "Player2": "Mochizuki S.",
    "Odds_A": 2.6,
    "Odds_B": 1.5,
    "Round": "1st Round"
  },
  {
    "Player1": "Evans D.",
    "Player2": "Clarke J.",
    "Odds_A": 1.18,
    "Odds_B": 5.0,
    "Round": "1st Round"
  },
  {
    "Player1": "Moutet C.",
    "Player2": "Comesana F.",
    "Odds_A": 1.45,
    "Odds_B": 2.8,
    "Round": "1st Round"
  },
  {
    "Player1": "Mpetshi G.",
    "Player2": "Fritz T.",
    "Odds_A": 5.5,
    "Odds_B": 1.15,
    "Round": "1st Round"
  },
  {
    "Player1": "Halys Q.",
    "Player2": "Holmgren A.",
    "Odds_A": 1.17,
    "Odds_B": 5.0,
    "Round": "1st Round"
  },
  {
    "Player1": "Machac T.",
    "Player2": "Dzumhur D.",
    "Odds_A": 1.2,
    "Odds_B": 4.5,
    "Round": "1st Round"
  },
  {
    "Player1": "Rinderknech A.",
    "Player2": "Zverev A.",
    "Odds_A": 7.5,
    "Odds_B": 1.08,
    "Round": "1st Round"
  },
  {
    "Player1": "Zhukayev B.",
    "Player2": "Cobolli F.",
    "Odds_A": 3.4,
    "Odds_B": 1.33,
    "Round": "1st Round"
  },
  {
    "Player1": "Mensik J.",
    "Player2": "Gaston H.",
    "Odds_A": 1.07,
    "Odds_B": 8.5,
    "Round": "1st Round"
  },
  {
    "Player1": "Cilic M.",
    "Player2": "Collignon R.",
    "Odds_A": 1.08,
    "Odds_B": 7.0,
    "Round": "1st Round"
  },
  {
    "Player1": "Bolt A.",
    "Player2": "Shelton B.",
    "Odds_A": 4.0,
    "Odds_B": 1.25,
    "Round": "1st Round"
  },
  {
    "Player1": "Shevchenko A.",
    "Player2": "Opelka R.",
    "Odds_A": 3.25,
    "Odds_B": 1.33,
    "Round": "1st Round"
  },
  {
    "Player1": "Humbert U.",
    "Player2": "Monfils G.",
    "Odds_A": 1.33,
    "Odds_B": 3.4,
    "Round": "1st Round"
  },
  {
    "Player1": "Kovacevic A.",
    "Player2": "Fucsovics M.",
    "Odds_A": 3.75,
    "Odds_B": 1.25,
    "Round": "1st Round"
  },
  {
    "Player1": "Nishioka Y.",
    "Player2": "Dimitrov G.",
    "Odds_A": 5.0,
    "Odds_B": 1.18,
    "Round": "1st Round"
  },
  {
    "Player1": "Goffin D.",
    "Player2": "Hijikata R.",
    "Odds_A": 2.38,
    "Odds_B": 1.6,
    "Round": "1st Round"
  },
  {
    "Player1": "Pinnington Jones J.",
    "Player2": "Etcheverry T.",
    "Odds_A": 2.75,
    "Odds_B": 1.45,
    "Round": "1st Round"
  },
  {
    "Player1": "Draper J.",
    "Player2": "Baez S.",
    "Odds_A": 1.02,
    "Odds_B": 15.0,
    "Round": "1st Round"
  },
  {
    "Player1": "Ugo Carabelli C.",
    "Player2": "Giron M.",
    "Odds_A": 9.0,
    "Odds_B": 1.06,
    "Round": "1st Round"
  },
  {
    "Player1": "Muller A.",
    "Player2": "Djokovic N.",
    "Odds_A": 15.0,
    "Odds_B": 1.02,
    "Round": "1st Round"
  },
  {
    "Player1": "Nakashima B.",
    "Player2": "Bu Y.",
    "Odds_A": 1.13,
    "Odds_B": 6.0,
    "Round": "1st Round"
  },
  {"Player1": "Tiafoe F.", "Player2": "Norrie C.", "Odds_A": 1.53, "Odds_B": 2.5, "Round": "2nd Round"},
  {"Player1": "Brooksby J.", "Player2": "Fonseca J.", "Odds_A": 2.8, "Odds_B": 1.44, "Round": "2nd Round"},
  {"Player1": "Mannarino A.", "Player2": "Royer V.", "Odds_A": 1.44, "Odds_B": 2.8, "Round": "2nd Round"},
  {"Player1": "Tien L.", "Player2": "Jarry N.", "Odds_A": 2.2, "Odds_B": 1.67, "Round": "2nd Round"},
  {"Player1": "Khachanov K.", "Player2": "Mochizuki S.", "Odds_A": 1.14, "Odds_B": 5.5, "Round": "2nd Round"},
  {"Player1": "Rublev A.", "Player2": "Harris L.", "Odds_A": 1.29, "Odds_B": 3.75, "Round": "2nd Round"},
  {"Player1": "Borges N.", "Player2": "Harris B.", "Odds_A": 1.53, "Odds_B": 2.5, "Round": "2nd Round"},
  {"Player1": "Tarvet O.", "Player2": "Alcaraz C.", "Odds_A": 12, "Odds_B": 1.01, "Round": "2nd Round"},
  {"Player1": "Bonzi B.", "Player2": "Thompson J.", "Odds_A": 1.53, "Odds_B": 2.5, "Round": "2nd Round"},
  {"Player1": "Bellucci M.", "Player2": "Lehecka J.", "Odds_A": 4.5, "Odds_B": 1.2, "Round": "2nd Round"},
  {"Player1": "Quinn E.", "Player2": "Majchrzak K.", "Odds_A": 1.62, "Odds_B": 2.2, "Round": "2nd Round"},
  {"Player1": "Fritz T.", "Player2": "Diallo G.", "Odds_A": 1.3, "Odds_B": 3.6, "Round": "2nd Round"},
  {"Player1": "Kecmanovic M.", "Player2": "De Jong J.", "Odds_A": 1.4, "Odds_B": 3, "Round": "2nd Round"},
  {"Player1": "Giron M.", "Player2": "Mensik J.", "Odds_A": 2.5, "Odds_B": 1.53, "Round": "2nd Round"},
  {"Player1": "Cazaux A.", "Player2": "De Minaur A.", "Odds_A": 8, "Odds_B": 1.08, "Round": "2nd Round"},
  {"Player1": "Cobolli F.", "Player2": "Pinnington Jones J.", "Odds_A": 1.36, "Odds_B": 3.2, "Round": "2nd Round"},
  {"Player1": "Moutet C.", "Player2": "Dimitrov G.", "Odds_A": 2.38, "Odds_B": 1.6, "Round": "2nd Round"},
  {"Player1": "Davidovich Fokina A.", "Player2": "Van De Zandschulp B.", "Odds_A": 1.3, "Odds_B": 3.6, "Round": "2nd Round"},
  {"Player1": "Marozsan F.", "Player2": "Munar J.", "Odds_A": 2, "Odds_B": 1.8, "Round": "2nd Round"},
  {"Player1": "Auger-Aliassime F.", "Player2": "Struff J.L.", "Odds_A": 1.33, "Odds_B": 3.25, "Round": "2nd Round"},
  {"Player1": "Rinderknech A.", "Player2": "Garin C.", "Odds_A": 1.67, "Odds_B": 2.25, "Round": "2nd Round"},
  {"Player1": "Djokovic N.", "Player2": "Evans D.", "Odds_A": 1.06, "Odds_B": 9, "Round": "2nd Round"},
  {"Player1": "Fery A.", "Player2": "Darderi L.", "Odds_A": 1.7, "Odds_B": 2.1, "Round": "2nd Round"},
  {"Player1": "Machac T.", "Player2": "Holmgren A.", "Odds_A": 1.15, "Odds_B": 5.5, "Round": "2nd Round"},
  {"Player1": "Martinez P.", "Player2": "Navone M.", "Odds_A": 2.63, "Odds_B": 1.5, "Round": "2nd Round"},
  {"Player1": "Nakashima B.", "Player2": "Opelka R.", "Odds_A": 1.4, "Odds_B": 3, "Round": "2nd Round"},
  {"Player1": "Ofner S.", "Player2": "Paul T.", "Odds_A": 5.5, "Odds_B": 1.15, "Round": "2nd Round"},
  {"Player1": "Draper J.", "Player2": "Cilic M.", "Odds_A": 1.12, "Odds_B": 6.5, "Round": "2nd Round"},
  {"Player1": "Sonego L.", "Player2": "Basilashvili N.", "Odds_A": 1.44, "Odds_B": 2.8, "Round": "2nd Round"},
  {"Player1": "Sinner J.", "Player2": "Vukic A.", "Odds_A": 1.01, "Odds_B": 17, "Round": "2nd Round"},
  {"Player1": "Monfils G.", "Player2": "Fucsovics M.", "Odds_A": 1.91, "Odds_B": 1.9, "Round": "2nd Round"},
  {"Player1": "Hijikata R.", "Player2": "Shelton B.", "Odds_A": 4.33, "Odds_B": 1.22, "Round": "2nd Round"},

  {"Player1": "Davidovich Fokina A.", "Player2": "Fritz T.", "Odds_A": 3.75, "Odds_B": 1.25, "Round": "3rd Round"},
  {"Player1": "Rublev A.", "Player2": "Mannarino A.", "Odds_A": 1.25, "Odds_B": 3.75, "Round": "3rd Round"},
  {"Player1": "Norrie C.", "Player2": "Bellucci M.", "Odds_A": 1.57, "Odds_B": 2.3, "Round": "3rd Round"},
  {"Player1": "Thompson J.", "Player2": "Darderi L.", "Odds_A": 1.67, "Odds_B": 2.1, "Round": "3rd Round"},
  {"Player1": "Rinderknech A.", "Player2": "Majchrzak K.", "Odds_A": 1.9, "Odds_B": 1.91, "Round": "3rd Round"},
  {"Player1": "Fonseca J.", "Player2": "Jarry N.", "Odds_A": 1.4, "Odds_B": 2.9, "Round": "3rd Round"},
  {"Player1": "Borges N.", "Player2": "Khachanov K.", "Odds_A": 3, "Odds_B": 1.4, "Round": "3rd Round"},
  {"Player1": "Struff J.L.", "Player2": "Alcaraz C.", "Odds_A": 11, "Odds_B": 1.04, "Round": "3rd Round"},
  {"Player1": "Cobolli F.", "Player2": "Mensik J.", "Odds_A": 2.5, "Odds_B": 1.53, "Round": "3rd Round"},
  {"Player1": "Martinez P.", "Player2": "Sinner J.", "Odds_A": 19, "Odds_B": 1.01, "Round": "3rd Round"},
  {"Player1": "Dimitrov G.", "Player2": "Ofner S.", "Odds_A": 1.29, "Odds_B": 3.75, "Round": "3rd Round"},
  {"Player1": "Sonego L.", "Player2": "Nakashima B.", "Odds_A": 2.6, "Odds_B": 1.5, "Round": "3rd Round"},
  {"Player1": "De Minaur A.", "Player2": "Holmgren A.", "Odds_A": 1.06, "Odds_B": 8, "Round": "3rd Round"},
  {"Player1": "Munar J.", "Player2": "Cilic M.", "Odds_A": 2.5, "Odds_B": 1.53, "Round": "3rd Round"},
  {"Player1": "Fucsovics M.", "Player2": "Shelton B.", "Odds_A": 3.6, "Odds_B": 1.3, "Round": "3rd Round"},
  {"Player1": "Kecmanovic M.", "Player2": "Djokovic N.", "Odds_A": 9.5, "Odds_B": 1.04, "Round": "3rd Round"},
  {"Player1": "Majchrzak K.", "Player2": "Khachanov K.", "Odds_A": 3.2, "Odds_B": 1.36, "Round": "4th Round"},
  {"Player1": "Thompson J.", "Player2": "Fritz T.", "Odds_A": 5.5, "Odds_B": 1.14, "Round": "4th Round"},
  {"Player1": "Norrie C.", "Player2": "Jarry N.", "Odds_A": 1.83, "Odds_B": 1.91, "Round": "4th Round"},
  {"Player1": "Rublev A.", "Player2": "Alcaraz C.", "Odds_A": 7, "Odds_B": 1.08, "Round": "4th Round"},
  {"Player1": "Cobolli F.", "Player2": "Cilic M.", "Odds_A": 1.83, "Odds_B": 2, "Round": "4th Round"},
  {"Player1": "Djokovic N.", "Player2": "De Minaur A.", "Odds_A": 1.17, "Odds_B": 5, "Round": "4th Round"},
  {"Player1": "Sonego L.", "Player2": "Shelton B.", "Odds_A": 3.75, "Odds_B": 1.29, "Round": "4th Round"},
  {"Player1": "Sinner J.", "Player2": "Dimitrov G.", "Odds_A": 1.03, "Odds_B": 13, "Round": "4th Round"},

  {"Player1": "Fritz T.", "Player2": "Khachanov K.", "Odds_A": 1.29, "Odds_B": 3.75, "Round": "Quarterfinals"},
  {"Player1": "Norrie C.", "Player2": "Alcaraz C.", "Odds_A": 9, "Odds_B": 1.06, "Round": "Quarterfinals"},
  {"Player1": "Sinner J.", "Player2": "Shelton B.", "Odds_A": 1.25, "Odds_B": 3.6, "Round": "Quarterfinals"},
  {"Player1": "Cobolli F.", "Player2": "Djokovic N.", "Odds_A": 7, "Odds_B": 1.1, "Round": "Quarterfinals"},

  {"Player1": "Fritz T.", "Player2": "Alcaraz C.", "Odds_A": 4.75, "Odds_B": 1.17, "Round": "Semifinals"},
  {"Player1": "Djokovic N.", "Player2": "Sinner J.", "Odds_A": 2.8, "Odds_B": 1.4, "Round": "Semifinals"},

  {"Player1": "Sinner J.", "Player2": "Alcaraz C.", "Odds_A": 1.83, "Odds_B": 2, "Round": "The Final"}
]

upcoming_df = pd.DataFrame(data)
upcoming_df['Surface'] = 'Grass'
upcoming_df['Series'] = 'Grand Slam'


import json

simulation_paris_2025()

# with open("joueurs.json", "w", encoding="utf-8") as f:
#     json.dump(list(player_stats.keys()), f, ensure_ascii=False, indent=4)

predict_match("Bonzi B.", ".", "Hard", 1.62, 2, 'Grand Slam', '3rd Round', ev_comparison=True, bankroll=17)

# !!!!!!!!!!!!!!!!!!!!!! avec min_ev = 0 et rmv_margin = False, on parie sur les même matchs qu'avec min_ev = 0.05 et rmv_margin = True

#69  2025-03-26      Eala A.            Swiatek I.    hard  0.587030      10.00  4.870300  54.114445  487.030008
# predict_match("Eala A.", "Swiatek I.", "Hard", 10, 1.06, "WTA1000", "Quaterfinals") # ATTENTION j'ai retiré 2025 et les tests du lgbmtennis.py
# IL Y A UN PROBLEME CAR IL SORT PAS LE MEME PARI QUE CE QU'ON POURRAIT ESPERER (on a les données jusqu'à fin 2024, peut être pour ça????) essayer jusqu'à ce match


# upcoming_X = build_upcoming_dataset(upcoming_df, player_stats).drop(columns=['Player1', 'Player2', 'Surface'])
# probas = model.predict_proba(upcoming_X)[:, 1]

# # Ajouter les résultats
# upcoming_df['Proba_P1_wins'] = probas

# pd.set_option('display.max_rows', None)  # Affiche toutes les lignes
# pd.set_option('display.max_columns', None)  # Affiche toutes les colonnes
# pd.set_option('display.width', None)  # Ne tronque pas horizontalement
# pd.set_option('display.max_colwidth', None)  # Affiche tout le contenu des cellules
# print(upcoming_df.head(64))

# plot_elo_evolution(match_df, "Alcaraz C.")