from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os, json, gzip
from functools import wraps

app = Flask(__name__)
CORS(app)

# Compression decorator for static files
def compressed_response(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        response = f(*args, **kwargs)
        if hasattr(response, 'headers'):
            response.headers['Cache-Control'] = 'public, max-age=86400'  # Cache for 24 hours
            response.headers['Vary'] = 'Accept-Encoding'
        return response
    return decorated_function

@app.route('/static/<path:filename>')
@compressed_response
def serve_static(filename):
    # Serve compressed version if available and client supports it
    if filename.endswith('.geojson'):
        gz_filename = filename + '.gz'
        gz_path = os.path.join('static', gz_filename)
        if os.path.exists(gz_path):
            response = send_from_directory('static', gz_filename)
            response.headers['Content-Encoding'] = 'gzip'
            response.headers['Content-Type'] = 'application/json'
            return response
    
    # Fallback to original file
    response = send_from_directory('static', filename)
    if filename.endswith('.geojson'):
        response.headers['Content-Type'] = 'application/json'
    return response

# ── Train model once on startup ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(os.path.join(BASE_DIR, 'calabarzon_rice_dataset.csv'))

data.rename(columns={
    'Avg_Temperature_C':    'avg_temp',
    'Rainfall_mm':          'rainfall_mm',
    'Soil_Moisture_%':      'soil_moisture',
    'Fertilizer_kg_per_ha': 'fertilizer_kg_ha',
    'Yield_tons_per_ha':    'yield_tons_ha'
}, inplace=True)

data = data.drop(['Year', 'Province'], axis=1)

# Generate synthetic disease severity based on humidity (soil_moisture) and rainfall
# High humidity + high rainfall promotes disease growth
soil_moisture_norm = (data['soil_moisture'] - data['soil_moisture'].min()) / (data['soil_moisture'].max() - data['soil_moisture'].min())
rainfall_norm = (data['rainfall_mm'] - data['rainfall_mm'].min()) / (data['rainfall_mm'].max() - data['rainfall_mm'].min())

# Calculate disease severity: combination of normalized humidity and rainfall (60% rainfall, 40% humidity)
# Scale to 0-5 range and round to integer
disease_severity_raw = (rainfall_norm * 0.6 + soil_moisture_norm * 0.4) * 5
data['disease_severity'] = np.round(disease_severity_raw).astype(int)
data['disease_severity'] = data['disease_severity'].clip(0, 5)  # Ensure values are between 0-5

# Apply disease penalty to yield: higher disease severity reduces yield
disease_penalty = 0.12  # Each unit of disease severity reduces yield by 12%
data['yield_tons_ha'] = data['yield_tons_ha'] * (1 - data['disease_severity'] * disease_penalty)

X = data[['avg_temp', 'rainfall_mm', 'soil_moisture', 'fertilizer_kg_ha', 'disease_severity']]
y = data['yield_tons_ha']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)
MSE  = float(mean_squared_error(y_test, y_pred_test))
R2   = float(r2_score(y_test, y_pred_test))
RMSE = float(np.sqrt(MSE))

FEATURE_IMPORTANCE = {
    f: float(imp)
    for f, imp in zip(X.columns, model.feature_importances_)
}

DATA_STATS = {
    'min_yield':  float(y.min()),
    'max_yield':  float(y.max()),
    'mean_yield': float(y.mean()),
    'mse':  MSE,
    'r2':   R2,
    'rmse': RMSE,
    'feature_importance': FEATURE_IMPORTANCE,
    'n_samples': len(data),
}

# Thresholds derived from dataset distribution
YIELD_LOW  = float(y.quantile(0.33))   # bottom third  → red
YIELD_HIGH = float(y.quantile(0.66))   # top third     → green


def classify_yield(y_val):
    if y_val >= YIELD_HIGH:
        return 'high'
    elif y_val >= YIELD_LOW:
        return 'moderate'
    else:
        return 'low'


def generate_tips(inputs, y_val, category):
    temp  = inputs['avg_temp']
    rain  = inputs['rainfall_mm']
    moist = inputs['soil_moisture']
    fert  = inputs['fertilizer_kg_ha']
    disease = inputs['disease_severity']
    tips  = []

    if category == 'high':
        tips.append({
            'icon': '✓',
            'type': 'success',
            'title': 'Excellent growing conditions',
            'body': f'Predicted yield of {y_val:.2f} t/ha is in the top tier. Maintain current practices.'
        })
    
    if category == 'moderate':
        tips.append({
            'icon': '📊',
            'type': 'warning',
            'title': 'Moderate yield potential',
            'body': f'Predicted yield of {y_val:.2f} t/ha is in the moderate range. Consider optimization opportunities below.'
        })

    if fert < 120:
        tips.append({
            'icon': '↑',
            'type': 'warning',
            'title': 'Increase fertilizer application',
            'body': f'Current rate is {fert:.0f} kg/ha. Historical data shows yields above 5.5 t/ha correlate with 150–200 kg/ha. Gradually increase to 150 kg/ha and monitor response.'
        })
    elif fert > 220:
        tips.append({
            'icon': '↓',
            'type': 'info',
            'title': 'Fertilizer rate is very high',
            'body': f'At {fert:.0f} kg/ha, you are above the dataset maximum. Diminishing returns may apply — consider splitting applications to improve absorption.'
        })

    if rain < 1800:
        tips.append({
            'icon': '💧',
            'type': 'warning',
            'title': 'Low rainfall — supplement irrigation',
            'body': f'Rainfall of {rain:.0f} mm is below the productive range (1800–2700 mm). Supplement with drip or flood irrigation, especially at tillering and heading stages.'
        })
    elif rain > 2600:
        tips.append({
            'icon': '⚠',
            'type': 'info',
            'title': 'High rainfall — watch for flooding',
            'body': f'At {rain:.0f} mm, excess water may cause nutrient leaching. Ensure proper field drainage and consider raised beds in low-lying areas.'
        })

    if moist < 50:
        tips.append({
            'icon': '🌱',
            'type': 'warning',
            'title': 'Improve soil moisture retention',
            'body': f'Soil moisture at {moist:.0f}% is low. Apply organic mulch or compost to improve water-holding capacity. Consider alternate wetting and drying (AWD) irrigation technique.'
        })
    elif moist > 75:
        tips.append({
            'icon': '✓',
            'type': 'success',
            'title': 'Soil moisture is optimal',
            'body': f'At {moist:.0f}%, moisture is in the high-productive range. Maintain this with consistent irrigation scheduling.'
        })

    if temp > 29:
        tips.append({
            'icon': '🌡',
            'type': 'warning',
            'title': 'High temperature stress risk',
            'body': f'Temperature of {temp:.1f}°C is above optimal (25–28°C). Schedule irrigation during early morning or evening to cool the canopy. Use heat-tolerant varieties like NSIC Rc 222.'
        })
    elif temp < 25:
        tips.append({
            'icon': '🌡',
            'type': 'info',
            'title': 'Low temperature may slow growth',
            'body': f'At {temp:.1f}°C, growth rate may be reduced. Use cold-tolerant upland varieties and ensure adequate potassium fertilization to improve stress tolerance.'
        })

    # Disease severity tips
    if disease == 0:
        tips.append({
            'icon': '🌿',
            'type': 'success',
            'title': 'No disease detected',
            'body': 'Excellent disease-free conditions. Continue with preventive measures and regular monitoring.'
        })
    elif disease == 1:
        tips.append({
            'icon': '⚠️',
            'type': 'info',
            'title': 'Very low disease pressure',
            'body': 'Minimal disease presence. Maintain current preventive practices and monitor for early symptoms.'
        })
    elif disease == 2:
        tips.append({
            'icon': '🔍',
            'type': 'warning',
            'title': 'Low disease pressure',
            'body': 'Early disease detected. Consider preventive fungicide application and improve field drainage.'
        })
    elif disease == 3:
        tips.append({
            'icon': '🦠',
            'type': 'warning',
            'title': 'Moderate disease pressure',
            'body': 'Significant disease presence. Apply targeted fungicides and remove infected plants to prevent spread.'
        })
    elif disease == 4:
        tips.append({
            'icon': '🚨',
            'type': 'danger',
            'title': 'High disease pressure',
            'body': 'Severe disease outbreak. Immediate intervention required with systemic fungicides and crop rotation.'
        })
    elif disease == 5:
        tips.append({
            'icon': '☠️',
            'type': 'danger',
            'title': 'Very high disease pressure',
            'body': 'Critical disease situation. Emergency measures needed - consider partial harvest and complete field treatment.'
        })

    # General disease management tips
    if disease >= 3:
        tips.append({
            'icon': '💊',
            'type': 'warning',
            'title': 'Disease management required',
            'body': f'Disease severity {disease}/5 requires immediate attention. Consult DA-CALABARZON for recommended fungicides.'
        })

    if category == 'low':
        tips.append({
            'icon': '📋',
            'type': 'danger',
            'title': 'Priority action: soil testing recommended',
            'body': 'Low predicted yield suggests multiple limiting factors. Conduct soil nutrient analysis to identify deficiencies. Contact DA-CALABARZON extension services for crop recovery programs.'
        })

    return tips


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', stats=json.dumps(DATA_STATS))


@app.route('/api/predict', methods=['POST'])
def predict():
    body = request.get_json(force=True)
    provinces = body.get('provinces', [])
    results = []

    for prov in provinces:
        inp = {
            'avg_temp':         float(prov.get('avg_temp',         27.5)),
            'rainfall_mm':      float(prov.get('rainfall_mm',      2100)),
            'soil_moisture':    float(prov.get('soil_moisture',    65)),
            'fertilizer_kg_ha': float(prov.get('fertilizer_kg_ha', 150)),
            'disease_severity': int(prov.get('disease_severity', 2)),
        }
        df    = pd.DataFrame([inp])
        y_val = float(model.predict(df)[0])
        cat   = classify_yield(y_val)
        tips  = generate_tips(inp, y_val, cat)

        results.append({
            'id':       prov.get('id'),
            'name':     prov.get('name'),
            'yield':    round(y_val, 3),
            'category': cat,
            'inputs':   inp,
            'tips':     tips,
        })

    return jsonify({
        'results': results,
        'thresholds': {'low': round(YIELD_LOW, 3), 'high': round(YIELD_HIGH, 3)},
        'model':   DATA_STATS,
    })


@app.route('/api/stats')
def stats():
    return jsonify(DATA_STATS)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
