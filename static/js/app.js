/* -----------------------------------------------------
   Crop Yield Prediction - Frontend Logic
   ----------------------------------------------------- */

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    loadDropdownData();
    loadModelComparison();
    loadEDAPlots();
});

// --- Tab Navigation ----------------------------------

function initTabs() {
    const tabs = document.querySelectorAll('.tab');
    const contents = document.querySelectorAll('.tab-content');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            contents.forEach(c => c.classList.remove('active'));

            tab.classList.add('active');
            document.getElementById(tab.dataset.tab).classList.add('active');
        });
    });
}

// --- Populate Dropdowns ------------------------------

async function loadDropdownData() {
    try {
        const res = await fetch('/api/dropdown-data');
        const data = await res.json();

        populateSelect('crop', data.Crop);
        populateSelect('season', data.Season);
        populateSelect('state', data.State);
    } catch (err) {
        console.error('Failed to load dropdown data:', err);
    }
}

function populateSelect(id, items) {
    const select = document.getElementById(id);
    items.forEach(item => {
        const option = document.createElement('option');
        option.value = item;
        option.textContent = item;
        select.appendChild(option);
    });
}

// --- Prediction Form ---------------------------------

document.getElementById('predict-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const btn = document.getElementById('btn-predict');
    const btnText = btn.querySelector('.btn-text');
    const btnLoader = btn.querySelector('.btn-loader');
    const resultBox = document.getElementById('result-box');
    const errorBox = document.getElementById('error-box');

    // Hide previous results
    resultBox.style.display = 'none';
    errorBox.style.display = 'none';

    // Show loading state
    btn.disabled = true;
    btnText.style.display = 'none';
    btnLoader.style.display = 'inline';

    const payload = {
        crop: document.getElementById('crop').value,
        season: document.getElementById('season').value,
        state: document.getElementById('state').value,
        crop_year: document.getElementById('crop_year').value,
        area: document.getElementById('area').value,
        annual_rainfall: document.getElementById('annual_rainfall').value,
        fertilizer: document.getElementById('fertilizer').value,
        pesticide: document.getElementById('pesticide').value,
        avg_temperature: document.getElementById('avg_temperature').value,
        max_temperature: document.getElementById('max_temperature').value,
        min_temperature: document.getElementById('min_temperature').value,
    };

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await res.json();

        if (data.success) {
            document.getElementById('result-value').textContent = data.predicted_yield.toFixed(4);
            resultBox.style.display = 'block';
            resultBox.scrollIntoView({ behavior: 'smooth', block: 'center' });
        } else {
            document.getElementById('error-msg').textContent = `Error: ${data.error}`;
            errorBox.style.display = 'block';
        }
    } catch (err) {
        document.getElementById('error-msg').textContent = `Network error: ${err.message}`;
        errorBox.style.display = 'block';
    } finally {
        btn.disabled = false;
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
    }
});

// --- Model Comparison Table --------------------------

async function loadModelComparison() {
    try {
        const res = await fetch('/api/model-info');
        const data = await res.json();
        const bestModel = data.best_model;
        const tbody = document.getElementById('comparison-body');

        const models = ['XGBoost', 'Random Forest', 'KNN'];
        models.forEach(name => {
            if (!data[name]) return;
            const m = data[name];
            const isBest = name === bestModel;

            const row = document.createElement('tr');
            if (isBest) row.style.backgroundColor = '#F0FAF0';

            row.innerHTML = `
                <td><strong>${name}</strong></td>
                <td>${m.R2_Score}</td>
                <td>${m.RMSE}</td>
                <td>${m.MAE}</td>
                <td>${isBest
                    ? '<span class="badge-best">Best</span>'
                    : '<span class="badge-other">-</span>'
                }</td>
            `;
            tbody.appendChild(row);
        });
    } catch (err) {
        console.error('Failed to load model comparison:', err);
    }
}

// --- EDA Plots ---------------------------------------

const PLOT_TITLES = {
    'yield_distribution.png': 'Distribution of Crop Yield',
    'top_crops_yield.png': 'Top 15 Crops by Average Yield',
    'season_yield.png': 'Average Yield by Season',
    'correlation_heatmap.png': 'Feature Correlation Heatmap',
    'state_yield.png': 'Top 15 States by Average Yield',
    'yield_vs_features.png': 'Yield vs Key Features',
    'yearly_trend.png': 'Average Yield Over Years',
    'model_comparison.png': 'Model Comparison'
};

async function loadEDAPlots() {
    try {
        const res = await fetch('/api/eda-plots');
        const plots = await res.json();
        const grid = document.getElementById('eda-grid');

        // Show in a specific order, excluding model_comparison (shown in its own tab)
        const order = [
            'yield_distribution.png',
            'top_crops_yield.png',
            'season_yield.png',
            'state_yield.png',
            'correlation_heatmap.png',
            'yield_vs_features.png',
            'yearly_trend.png'
        ];

        const sortedPlots = order.filter(p => plots.includes(p));

        sortedPlots.forEach(filename => {
            const item = document.createElement('div');
            item.className = 'eda-item';
            item.innerHTML = `
                <img src="/eda_plots/${filename}" alt="${PLOT_TITLES[filename] || filename}" loading="lazy">
                <p>${PLOT_TITLES[filename] || filename}</p>
            `;
            grid.appendChild(item);
        });
    } catch (err) {
        console.error('Failed to load EDA plots:', err);
    }
}
