import requests
from predictions import predict_match

key2 = "15023aab03mshe741c694d87ba09p1d25c7jsn2831ec352181"
key1 = "30c4f58ed3msh2c671728169a68dp10d40djsneaf6b1a69049"
key = "76ec2fb838msh5b33f1561f7aa0ep11c67cjsn07f16a9259dd"
key3 = "f211a17182msh7eda8c94e801a60p1ecd31jsn4a6662c55314"

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

def getPredictionsOnDay(day, month, year, bankroll):
    matchPredictions = []
    try:
        url = f"https://tennisapi1.p.rapidapi.com/api/tennis/category/3/events/{day}/{month}/{year}"
        headers = {
                    "x-rapidapi-host": "tennisapi1.p.rapidapi.com",
                    "x-rapidapi-key": key
                }

        r = requests.get(url, headers=headers)
        data = r.json()

        urlW = f"https://tennisapi1.p.rapidapi.com/api/tennis/category/6/events/{day}/{month}/{year}"
        headersW = {
                    "x-rapidapi-host": "tennisapi1.p.rapidapi.com",
                    "x-rapidapi-key": key
                }

        rW = requests.get(urlW, headers=headersW)
        dataW = rW.json()
        # if data["message"]:
        #     print(data["message"])
        #     exit(1)

        for match in data["events"] + dataW["events"]:
            type = match["eventFilters"]["category"][0]
            status = match["status"]["type"]

            if type != "singles" or status != "notstarted":
                continue

            idMatch = match["id"]
            playerA = format_player(match["homeTeam"]["name"])
            playerB = format_player(match["awayTeam"]["name"])
            roundName = match["roundInfo"]["name"]
            groundType = surfaceCategories[match["groundType"].split(" ")[0]]
            tournamentLevel = tournamentCategories[match["eventFilters"]["tournament"][0]]

            urlOdds = f"https://tennisapi1.p.rapidapi.com/api/tennis/event/{idMatch}/odds/1/featured"
            headersOdds = {
                        "x-rapidapi-host": "tennisapi1.p.rapidapi.com",
                        "x-rapidapi-key": key
                    }
            
            rOdds = requests.get(urlOdds, headers=headersOdds)
            dataOdds = rOdds.json()
            choices = dataOdds["featured"]["default"]["choices"]
            oddA = round(1.0 + eval(choices[0]["fractionalValue"]), 2)
            oddB = round(1.0 + eval(choices[1]["fractionalValue"]), 2)

            print(playerA, "vs", playerB, "[", tournamentLevel, ",", roundName, "on", groundType, "] Cotes :", oddA, "|", oddB)
            matchPredictions.append(predict_match(playerA, playerB, groundType, oddA, oddB, tournamentLevel, roundName, bankroll=bankroll))
            print("append")

        # print(matchPredictions)
    except Exception:
        print(data)
    finally:
        matchPredictions.append({"Error": "Problem in main loop"})
        return matchPredictions
        


getPredictionsOnDay(30, 1, 2026, 10)