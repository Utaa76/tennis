const today = new Date().toISOString().split('T')[0];
document.getElementById('match-date').value = today;
console.log(today);

function createInfo(parent, infoName, value) {
    div = document.createElement('div');
    div.classList.add("info");
    ph = document.createElement('p');
    ph.classList.add("placeholder");
    ph.textContent = infoName;
    val = document.createElement('p');
    val.textContent = value;

    div.appendChild(ph);
    div.appendChild(val);
    
    parent.appendChild(div);
}

function renderBets(data) {
    console.log(data);
    const divBets = document.getElementById('bets');
    for (bet of data) {
        divBet = document.createElement('div');
        divBet.classList.add('bet');
        divBets.appendChild(divBet);

        match = document.createElement('h3');
        match.textContent = bet.match;

        divInfo = document.createElement('div');
        divInfo.classList.add('infos');

        createInfo(divInfo, "winner", bet.winner);
        createInfo(divInfo, "cote", bet.cote);
        createInfo(divInfo, "mise", (bet.mise).toFixed(2));
        createInfo(divInfo, "gain", (bet.gain_attendu).toFixed(2));

        divBet.appendChild(match);
        divBet.appendChild(divInfo);
    }
}

formDate = document.getElementById('form-date')
formDate.addEventListener('submit', (e) => {
    e.preventDefault();
    matchDate = document.getElementById('match-date')
    dateString = matchDate.value.split("-");
    year  = dateString[0];
    month = dateString[1];
    day   = dateString[2];

    bankroll = Number(document.getElementById('bankroll-input').value).toFixed(2);

    const data = {
        day: Number(day),
        month: Number(month),
        year: Number(year),
        bankroll: bankroll
    }

    const loader = document.getElementById("loader");
    loader.style.display = 'flex';

    fetch('http://127.0.0.1:8000/getPredictions', {
    // fetch('https://tennis-8gw3.onrender.com/getPredictions', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(data)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error("HTTP Error " + response.status);
        }
        return response.json();
    })
    .then(data => {
        console.log(data);
        renderBets(data);
    })
    .finally(() => {
        loader.style.display = 'none';
    })
});

dataTest = [
    {
        match: "Moi vs Toi",
        winner: "Moi",
        cote: 1.21,
        mise: 100.2,
        gain_attendu: 1.21*100.2
    },
    {
        match: "Moissonneuse vs Toiture",
        winner: "Toiture",
        cote: 21,
        mise: 10.2,
        gain_attendu: 21*10.2
    }
];