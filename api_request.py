import requests
from predictions import predict_match
import traceback

keys = ["15023aab03mshe741c694d87ba09p1d25c7jsn2831ec352181",
        "30c4f58ed3msh2c671728169a68dp10d40djsneaf6b1a69049",
        "76ec2fb838msh5b33f1561f7aa0ep11c67cjsn07f16a9259dd",
        "f211a17182msh7eda8c94e801a60p1ecd31jsn4a6662c55314",
        "ceeb7f0ca0msh3e6f0385a2f4479p1ef292jsn4df06447b071",
        "3dec61f071msh23bff88624e7fb3p17f627jsn4085dba9c965",
        "5944875573mshde49dacc57eb914p124939jsn15263fc9fe6e"
        ]


NAME_FIXES = {
    "Felix Auger Aliassime": "Auger-Aliassime F.",
    "Alex de Minaur": "De Minaur A.",
    "Learner Tien": "Tien L.",
    "Carlos Alcaraz": "Alcaraz C.",
    "Alexander Zverev": "Zverev A.",
    "Lorenzo Musetti": "Musetti L.",
    "Novak Djokovic": "Djokovic N.",
    "Ben Shelton": "Shelton B.",
    "Jannik Sinner": "Sinner J.",
    "Pablo Carreño Busta": "Carreno Busta P.",
    "Botic Van de Zandschulp": "Van De Zandschulp B.",
    "Martin Damm Jr": "Damm M.",
    "Zeynep Sönmez": "Sonmez Z.",
}

def format_player(name):
    if name in NAME_FIXES:
        return NAME_FIXES[name]

    print("name not in NAME_FIXES", name)
    parts = name.split()
    first = parts[0]
    last = " ".join(parts[1:])
    return f"{last} {first[0]}."


tournamentCategories = {
    "grand-slam": "Grand Slam",
    "p250": "ATP250",
    "p500": "ATP500",
    "p1000": "Masters 1000"
}

surfaceCategories = {
    "Hardcourt": "Hard",
    "Red clay": "Clay",
    "Grass": "Grass"
}

roundCategories = {
    "Round of 128": "1st Round",
    "Round of 64": "2nd Round",
    "Round of 32": "3rd Round",
    "Round of 16": "4th Round",
    "Quarterfinals": "Quarterfinals",
    "Semifinals": "Semifinals",
    "Final": "The Final"
}

def hasRequestFailed(json):
    if "message" in json:
        print(json["message"])
        return True
    
    return False


def getPredictionsOnDay(day, month, year, bankroll, keyId=0):
    matchPredictions = []

    if keyId >= len(keys):
        print("Toutes les clés API ont été épuisées !")
        return [{"Error": "All API keys exhausted"}]

    key = keys[keyId]

    try:
        # --- Singles ---
        url = f"https://tennisapi1.p.rapidapi.com/api/tennis/category/3/events/{day}/{month}/{year}"
        headers = {
            "x-rapidapi-host": "tennisapi1.p.rapidapi.com",
            "x-rapidapi-key": key
        }

        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        try:
            data = r.json()
        except ValueError:
            print(f"JSONDecodeError avec la clé {key}, essai avec la suivante...")
            return getPredictionsOnDay(day, month, year, bankroll, keyId + 1)

        if "message" in data:
            print(f"Erreur API avec la clé {key} : {data['message']}")
            return getPredictionsOnDay(day, month, year, bankroll, keyId + 1)

        # --- Womens ---
        urlW = f"https://tennisapi1.p.rapidapi.com/api/tennis/category/6/events/{day}/{month}/{year}"
        rW = requests.get(urlW, headers=headers, timeout=10)
        rW.raise_for_status()
        try:
            dataW = rW.json()
        except ValueError:
            print(f"JSONDecodeError (Womens) avec la clé {key}, essai avec la suivante...")
            return getPredictionsOnDay(day, month, year, bankroll, keyId + 1)

        if "message" in dataW:
            print(f"Erreur API (Womens) avec la clé {key} : {dataW['message']}")
            return getPredictionsOnDay(day, month, year, bankroll, keyId + 1)

        # --- Boucle sur les matchs ---
        for match in data["events"] + dataW["events"]:
            match_type = match["eventFilters"]["category"][0]
            status = match["status"]["type"]

            if match_type != "singles" or status != "notstarted":
                continue

            # Récupération des infos
            idMatch = match["id"]
            playerA = format_player(match["homeTeam"]["name"])
            playerB = format_player(match["awayTeam"]["name"])
            roundName = match["roundInfo"]["name"]
            groundType = surfaceCategories[match["groundType"].split(" ")[0]]
            tournamentLevel = tournamentCategories[match["eventFilters"]["tournament"][0]]

            # --- Récupération des cotes ---
            while True:
                urlOdds = f"https://tennisapi1.p.rapidapi.com/api/tennis/event/{idMatch}/odds/1/featured"
                rOdds = requests.get(urlOdds, headers=headers, timeout=10)
                try:
                    dataOdds = rOdds.json()
                except ValueError:
                    print(f"JSONDecodeError (Odds) avec la clé {key}, essai avec la suivante...")
                    keyId += 1
                    if keyId >= len(keys):
                        return matchPredictions
                    key = keys[keyId]
                    headers["x-rapidapi-key"] = key
                    continue

                if "message" in dataOdds:
                    print(f"Erreur API (Odds) avec la clé {key} : {dataOdds['message']}")
                    keyId += 1
                    if keyId >= len(keys):
                        return matchPredictions
                    key = keys[keyId]
                    headers["x-rapidapi-key"] = key
                    continue

                break  # tout ok

            # Calcul des cotes
            choices = dataOdds["featured"]["default"]["choices"]
            oddA = round(1.0 + eval(choices[0]["fractionalValue"]), 2)
            oddB = round(1.0 + eval(choices[1]["fractionalValue"]), 2)

            print(playerA, "vs", playerB, "[", tournamentLevel, ",", roundName, "on", groundType, "] Cotes :", oddA, "|", oddB)
            matchPredictions.append(
                predict_match(playerA, playerB, groundType, oddA, oddB, tournamentLevel, roundName, bankroll=bankroll)
            )

    except requests.exceptions.RequestException as e:
        print(f"Erreur requête HTTP avec la clé {key} : {e}")
        return getPredictionsOnDay(day, month, year, bankroll, keyId + 1)
    except Exception as e:
        print(f"Erreur inattendue : {e}")
        traceback.print_exc()
        matchPredictions.append({"Error": "Problem in main loop"})

    return matchPredictions

        


# getPredictionsOnDay(30, 1, 2026, 10)