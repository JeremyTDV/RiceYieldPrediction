# AGRI-GIS CALABARZON — Rice Yield Prediction System

A full-stack web GIS application that uses a trained **Random Forest** model to predict rice yield across the 5 provinces of CALABARZON (Region IV-A, Philippines). After entering environmental and input parameters, the map highlights each province **green** (high yield), **yellow** (moderate), or **red** (low yield), with actionable improvement tips.

---

## Project Structure

```
rice_app/
├── app.py                  # Flask backend — trains model, serves API
├── combined_rice_data.csv  # Training dataset (1985–2024)
├── requirements.txt        # Python dependencies
├── Procfile                # For Heroku / Railway deployment
├── templates/
│   └── index.html          # Full GIS frontend (Leaflet + custom UI)
└── README.md
```

---

## Running Locally

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the server
```bash
python app.py
```

### 3. Open in browser
```
http://localhost:5000
```

The model trains automatically on startup (~2 seconds). No pre-training step needed.

---

## Deploying to the Web

### Option A — Render (recommended, free tier)

1. Push the `rice_app/` folder to a GitHub repository
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Set these values:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Environment:** Python 3
5. Click **Deploy** — your app will be live at `https://your-app.onrender.com`

### Option B — Railway

1. Push to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Railway auto-detects the `Procfile` and deploys
4. Live at `https://your-app.up.railway.app`

### Option C — Heroku

```bash
heroku create your-app-name
git push heroku main
heroku open
```

### Option D — PythonAnywhere (free)

1. Upload the project folder via Files tab
2. Create a new Web App → Flask → Python 3.11
3. Set **Source code** path to your upload directory
4. Set **WSGI file** to point to `app.py`
5. Install requirements via Bash console:
   ```bash
   pip install -r requirements.txt --user
   ```
6. Reload the app

---

## API Endpoints

### `POST /api/predict`
Accepts province-level environmental inputs, returns yield predictions with tips.

**Request body:**
```json
{
  "provinces": [
    {
      "id": "cavite",
      "name": "Cavite",
      "avg_temp": 27.5,
      "rainfall_mm": 2100,
      "soil_moisture": 65,
      "fertilizer_kg_ha": 150
    }
  ]
}
```

**Response:**
```json
{
  "results": [
    {
      "id": "cavite",
      "name": "Cavite",
      "yield": 5.23,
      "category": "high",
      "inputs": { ... },
      "tips": [
        {
          "icon": "✓",
          "type": "success",
          "title": "Excellent growing conditions",
          "body": "..."
        }
      ]
    }
  ],
  "thresholds": { "low": 4.21, "high": 5.44 },
  "model": { "r2": 0.874, "mse": 0.092, "rmse": 0.303, ... }
}
```

### `GET /api/stats`
Returns model performance metrics and feature importances.

---

## How the Yield Classification Works

Thresholds are computed dynamically from the dataset's **33rd and 66th percentiles**:

| Category | Condition          | Map color |
|----------|--------------------|-----------|
| **High** | ≥ 66th percentile  | 🟢 Green  |
| **Moderate** | 33rd–66th pct  | 🟡 Yellow |
| **Low** | < 33rd percentile  | 🔴 Red    |

---

## Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | Random Forest Regressor |
| Trees | 100 (`n_estimators=100`) |
| Train/Test split | 80% / 20% |
| Random state | 42 |
| Features | Temperature, Rainfall, Soil Moisture, Fertilizer |
| Target | Yield (tons/hectare) |

---

## Feature Importance (approximate)

| Feature | Importance |
|---------|-----------|
| Fertilizer (kg/ha) | ~52% |
| Rainfall (mm) | ~22% |
| Soil Moisture (%) | ~15% |
| Avg Temperature (°C) | ~11% |

---

## Data Source

Historical rice production data for Calabarzon region (1985–2024), compiled from DA-CALABARZON agricultural statistics.

---

## Tech Stack

- **Backend:** Python, Flask, scikit-learn, pandas, NumPy
- **Frontend:** Vanilla JS, Leaflet.js, CartoDB basemap tiles
- **Deployment:** Gunicorn WSGI server
