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

const form = document.getElementById('betForm')
const resultDiv = document.getElementById('result')

console.log("bien reçu")

form.addEventListener('submit', async (e) => {
	e.preventDefault();

	const formData = new FormData(form);
	const data = {};
	formData.forEach((value, key) => {
		if (key === 'cote_A' || key === 'cote_B' || key === 'bankroll' || key === 'min_ev') {
			data[key] = parseFloat(value);
		} else {
			data[key] = value;
		}
	});

	resultDiv.textcontent = 'Chargement...';

	try {
		const response = await fetch('https://tennis-8gw3.onrender.com/predict', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(data)
		});

		if (!response.ok) {
			throw new Error('Erreur réseau ou serveur')
		}

		const json = await response.json();

		if (json.message) {
			resultDiv.textContent = json.message;
		} else {
			resultDiv.innerHTML = `
				<strong>Match :</strong> ${json.match}<br/>
				<strong>Pari recommandé :</strong> ${json.winner}<br/>
				<strong>Mise :</strong> ${(json.mise).toFixed(2)} €<br/>
				<strong>Gain attendu :</strong> ${(json.gain_attendu).toFixed(2)} €
			`;
		};
	} catch (err) {
		resultDiv.textcontent = 'Erreur : ' + err.message;
	}
});