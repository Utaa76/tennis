fetch("/static/joueurs.json")
	.then(res => res.json())
	.then(joueurs => {
		const datalist = document.getElementById("joueurs");
		joueurs.forEach(joueur => {
			const option = document.createElement("option");
			option.value = joueur;
			datalist.appendChild(option);
		});
		console.log(datalist)
	})
	.catch(error => console.error("Erreur de chargement des joueurs :", error));

const selectLevel = document.getElementById("level_name")
const selectRound = document.getElementById("round_name")

const surfaceSelectors = document.querySelectorAll(".surface-selector");

function updateColor() {
	if (!selectLevel.value) {
		selectLevel.classList.remove("active");
	} else {
		selectLevel.classList.add("active");
	}

	if (!selectRound.value) {
		selectRound.classList.remove("active");
	} else {
		selectRound.classList.add("active");
	}
}

function selectSurface(self) {
	surfaceSelectors.forEach(surfaceSelector => {
		surfaceSelector.classList.remove("selected");
	});

	self.classList.add("selected");
}

updateColor();

selectLevel.addEventListener("change", updateColor);
selectRound.addEventListener("change", updateColor);

surfaceSelectors.forEach(surfaceSelector => {
	surfaceSelector.addEventListener("click", () => selectSurface(surfaceSelector));
});

const form = document.getElementById('betForm');
const resultDiv = document.getElementById('result');
const joueur1 = document.getElementById('joueur1');
const joueur2 = document.getElementById('joueur2');
const infosBet = document.getElementById('infos-bet');

form.addEventListener('submit', async (e) => {
    e.preventDefault(); // Empêche le rechargement de la page

	form.classList.add("hidden");
	resultDiv.classList.remove("hidden");

    // Construire l'objet data à partir du formulaire
    const data = {
        A: document.getElementById('A').value,
        B: document.getElementById('B').value,
        cote_A: parseFloat(document.getElementById('cote_A').value),
        cote_B: parseFloat(document.getElementById('cote_B').value),
        level_name: document.getElementById('level_name').value,
        round_name: document.getElementById('round_name').value,
        bankroll: parseFloat(document.getElementById('bankroll').value),
        min_ev: parseFloat(document.getElementById('min_ev').value),
        surface: getSelectedSurface()
    };

    try {
        const response = await fetch('https://tennis-8gw3.onrender.com/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error(`Erreur serveur (${response.status})`);
        }

        const json = await response.json();

        if (json.message) {
            infosBet.textContent = json.message;
        } else {
			joueur1.textContent = json.joueur1;
			joueur2.textContent = json.joueur2;

			if (json.joueur1 === json.winner) {
				joueur1.classList.add("winner");
			} else {
				joueur2.classList.add("winner");
			}


            infosBet.innerHTML = `
                <strong>Pari recommandé :</strong> ${json.winner}<br/>
                <strong>Mise :</strong> ${(json.mise).toFixed(2)} €<br/>
                <strong>Gain attendu :</strong> ${(json.gain_attendu).toFixed(2)} € <br/>
                <strong>Probabilité :</strong> ${(json.probability*100).toFixed(2)} %
            `;
        }
    } catch (err) {
        resultDiv.textContent = 'Erreur : ' + err.message;
    }
});

// Fonction pour récupérer la surface sélectionnée
function getSelectedSurface() {
    const surfaces = document.querySelectorAll('.surface-selector');
    for (const s of surfaces) {
        if (s.classList.contains('selected')) {
            console.log("Surface : " + s.textContent);
            return s.textContent;
        }
    }
    return null; // ou une valeur par défaut si nécessaire
}

// Exemple : ajout de la sélection de surface
const surfaces = document.querySelectorAll('.surface-selector');
surfaces.forEach(s => {
    s.addEventListener('click', () => {
        surfaces.forEach(x => x.classList.remove('selected'));
        s.classList.add('selected');
    });
});
