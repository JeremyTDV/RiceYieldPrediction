# AGRI-GIS CALABARZON — Rice Yield Prediction System

A full-stack web GIS application that uses a trained **Random Forest** model to predict rice yield across the 5 provinces of CALABARZON (Region IV-A, Philippines). After entering environmental and input parameters, the map highlights each province **green** (high yield), **yellow** (moderate), or **red** (low yield), with actionable improvement tips.

---

## Project Structure

```
rice_app/
├── app.py                      # Flask backend — trains model, serves API, auth, admin
├── calabarzon_rice_dataset.csv # Training dataset (processed)
├── combined_rice_data.csv      # Source historical data (1985–2024)
├── compress_geojson.py         # GeoJSON compression utility
├── requirements.txt            # Python dependencies
├── Procfile                    # For Render / Railway deployment
├── static/
│   └── calabarzon.geojson[.gz] # CALABARZON map (gzip-compressed)
├── templates/
│   ├── index.html              # Main GIS dashboard (Leaflet + custom UI)
│   └── login.html              # Login page
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

You will be redirected to a login page. The model trains automatically on startup (~2 seconds).

**Demo credentials:**
- Admin: `admin` / `admin123`
- User: `user` / `user123`

---

## Deploying to the Web

### Option A — Render (recommended, free tier)

1. Push the folder to a GitHub repository
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

### Authentication

#### `POST /api/login`
Authenticate user with credentials.

**Request body:**
```json
{
  "username": "admin",
  "password": "admin123"
}
```

#### `POST /api/logout`
Log out the current user.

#### `GET /api/me`
Check logged-in status and user details.

### Admin Endpoints

#### `GET /api/admin/rice-areas`
Retrieve current rice cultivation areas (hectares) per province.

#### `POST /api/admin/rice-areas`
Update rice cultivation areas for one or more provinces.

**Request body:**
```json
{
  "cavite": 15000,
  "laguna": 32000
}
```

#### `POST /api/admin/rice-areas/reset`
Reset rice areas to default values.

### Prediction Endpoints

#### `POST /api/predict`
Accepts province-level environmental inputs, returns yield predictions with tips. Requires login.

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
      "fertilizer_kg_ha": 150,
      "disease_severity": 2
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
      "rice_area": 14000,
      "total_yield_tons": 73220,
      "category": "high",
      "inputs": { ... },
      "tips": [
        {
          "icon": "✓",
          "type": "success",
          "title": "Excellent growing conditions",
          "body": "Predicted yield of 5.23 t/ha is in the top tier. Maintain current practices."
        }
      ]
    }
  ],
  "thresholds": { "low": 4.21, "high": 5.44 },
  "model": { "r2": 0.874, "mse": 0.092, "rmse": 0.303, ... }
}
```

#### `GET /api/stats`
Returns model performance metrics, feature importances, and dataset statistics. Requires login.

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
| Estimators | 200 |
| Max Depth | 15 |
| Min Samples Split | 5 |
| Min Samples Leaf | 2 |
| Train/Test split | 80% / 20% |
| Random state | 42 |
| Features | Avg Temperature, Rainfall, Soil Moisture, Fertilizer, Rice Area, Disease Severity |
| Target | Yield (tons/hectare) |

---

## Features & Disease Severity

### Input Features

| Feature | Description | Unit |
|---------|-------------|------|
| Avg Temperature | Mean temperature during growing season | °C |
| Rainfall | Cumulative precipitation during season | mm |
| Soil Moisture | Volumetric water content in soil | % |
| Fertilizer | Application rate | kg/ha |
| Rice Area | Cultivated area for the province | hectares |
| Disease Severity | Calculated from rainfall & soil moisture | 0–5 scale |

### Disease Severity Calculation

Disease severity is derived from environmental conditions:
- **Formula:** `(rainfall_norm × 0.6 + soil_moisture_norm × 0.4) × 5`, then rounded to integer 0–5
- **Impact:** Yield is penalized by 12% per severity level
- **0 = No disease** → minimal penalty  
- **5 = Critical outbreak** → severe yield reduction (60% penalty)

### Feature Importance (from model training)

Based on the trained Random Forest model, the following features ranked by importance:

| Feature | Importance | Percentage |
|---------|-----------|-----------|
| Fertilizer (kg/ha) | 0.4992 | 49.92% |
| Disease Severity | 0.2331 | 23.31% |
| Soil Moisture (%) | 0.1452 | 14.52% |
| Rainfall (mm) | 0.0769 | 7.69% |
| Avg Temperature (°C) | 0.0359 | 3.59% |
| Rice Area (ha) | 0.0096 | 0.96% |

Fertilizer application is the most predictive factor for rice yield, followed by disease severity. Environmental conditions (rainfall, soil moisture, temperature) have significant but secondary influence. Rice cultivated area has minimal impact on per-hectare yield.

---

## Data Source

Historical rice production data for Calabarzon region (1985–2024), compiled from DA-CALABARZON agricultural statistics.

---

## Tech Stack

- **Backend:** Python 3, Flask, scikit-learn, pandas, NumPy, Flask-CORS
- **Frontend:** Vanilla JavaScript, Leaflet.js, CartoDB basemap tiles
- **Deployment:** Gunicorn WSGI server
- **Data:** GeoJSON (gzip-compressed for performance)
- **Authentication:** Session-based (Flask sessions)
