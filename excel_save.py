import pandas as pd
import os

def ajouter_matchs_excel(nouvelles_lignes, fichier="paris_tennis.xlsx"):
    # Colonnes souhaitées
    colonnes = ['match', 'winner', 'cote', 'mise', 'gain_attendu', 'resultat', 'gain_net']

    # Créer DataFrame pour les nouvelles lignes
    df_nouvelles = pd.DataFrame(nouvelles_lignes)[['match', 'winner', 'cote', 'mise', 'gain_attendu']]
    df_nouvelles['resultat'] = ""
    df_nouvelles['gain_net'] = df_nouvelles.apply(
        lambda row: f"=IF(F{row.name+2}=\"W\", E{row.name+2}, -D{row.name+2})", axis=1
    )

    if os.path.exists(fichier):
        # Lire l'existant
        df_exist = pd.read_excel(fichier)
        # Concaténer
        df_final = pd.concat([df_exist, df_nouvelles], ignore_index=True)
    else:
        df_final = df_nouvelles

    # Écrire ou réécrire le fichier Excel
    df_final.to_excel(fichier, index=False, engine='openpyxl')
    print(f"{len(df_nouvelles)} matchs ajoutés dans {fichier}")