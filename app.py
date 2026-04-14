from flask import Flask, request, jsonify, render_template, send_from_directory, session, redirect, url_for
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os, json, gzip
from functools import wraps

app = Flask(__name__)
app.secret_key = 'agrigis-calabarzon-secret-2024'
CORS(app)

# ── User accounts ─────────────────────────────────────────────────────────────
# In production, use a real database. For thesis demo, hardcoded is fine.
USERS = {
    'admin': {'password': 'admin123', 'role': 'admin', 'name': 'Administrator'},
    'user':  {'password': 'user123',  'role': 'user',  'name': 'DA Technician'},
}

# ── Mutable constants (admin can change these) ────────────────────────────────
# Rice area harvested per province (ha) — PSA/NAMRIA source
RICE_AREAS = {
    'cavite':   14000,
    'laguna':   30000,
    'batangas':  8000,
    'rizal':     5800,
    'quezon':   43000,
}

# ── Auth helpers ──────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session:
            if request.is_json:
                return jsonify({'error': 'Unauthorized', 'redirect': '/login'}), 401
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session:
            return jsonify({'error': 'Unauthorized'}), 401
        if session.get('role') != 'admin':
            return jsonify({'error': 'Admin access required'}), 403
        return f(*args, **kwargs)
    return decorated

# ── Compression decorator ─────────────────────────────────────────────────────
def compressed_response(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        response = f(*args, **kwargs)
        if hasattr(response, 'headers'):
            response.headers['Cache-Control'] = 'public, max-age=86400'
            response.headers['Vary'] = 'Accept-Encoding'
        return response
    return decorated_function

@app.route('/static/<path:filename>')
@compressed_response
def serve_static(filename):
    if filename.endswith('.geojson'):
        gz_filename = filename + '.gz'
        gz_path = os.path.join('static', gz_filename)
        if os.path.exists(gz_path):
            response = send_from_directory('static', gz_filename)
            response.headers['Content-Encoding'] = 'gzip'
            response.headers['Content-Type'] = 'application/json'
            return response
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

soil_moisture_norm = (data['soil_moisture'] - data['soil_moisture'].min()) / (data['soil_moisture'].max() - data['soil_moisture'].min())
rainfall_norm = (data['rainfall_mm'] - data['rainfall_mm'].min()) / (data['rainfall_mm'].max() - data['rainfall_mm'].min())
disease_severity_raw = (rainfall_norm * 0.6 + soil_moisture_norm * 0.4) * 5
data['disease_severity'] = np.round(disease_severity_raw).astype(int).clip(0, 5)
disease_penalty = 0.12
data['yield_tons_ha'] = data['yield_tons_ha'] * (1 - data['disease_severity'] * disease_penalty)

X = data[['avg_temp', 'rainfall_mm', 'soil_moisture', 'fertilizer_kg_ha', 'rice_area', 'disease_severity']]
y = data['yield_tons_ha']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(
    n_estimators=200, max_depth=15, min_samples_split=5,
    min_samples_leaf=2, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)
MSE  = float(mean_squared_error(y_test, y_pred_test))
R2   = float(r2_score(y_test, y_pred_test))
RMSE = float(np.sqrt(MSE))

FEATURE_IMPORTANCE = {f: float(imp) for f, imp in zip(X.columns, model.feature_importances_)}

DATA_STATS = {
    'min_yield': float(y.min()), 'max_yield': float(y.max()),
    'mean_yield': float(y.mean()), 'mse': MSE, 'r2': R2, 'rmse': RMSE,
    'feature_importance': FEATURE_IMPORTANCE, 'n_samples': len(data),
}

YIELD_LOW  = float(y.quantile(0.33))
YIELD_HIGH = float(y.quantile(0.66))


def classify_yield(y_val):
    if y_val >= YIELD_HIGH: return 'high'
    elif y_val >= YIELD_LOW: return 'moderate'
    else: return 'low'


def generate_tips(inputs, y_val, category):
    temp    = inputs['avg_temp']
    rain    = inputs['rainfall_mm']
    moist   = inputs['soil_moisture']
    fert    = inputs['fertilizer_kg_ha']
    disease = inputs['disease_severity']
    tips    = []

    if category == 'high':
        tips.append({'icon':'✓','type':'success','title':'Excellent growing conditions',
            'body':f'Predicted yield of {y_val:.2f} t/ha is in the top tier. Maintain current practices.'})
    if category == 'moderate':
        tips.append({'icon':'📊','type':'warning','title':'Moderate yield potential',
            'body':f'Predicted yield of {y_val:.2f} t/ha is in the moderate range. Consider optimization opportunities below.'})
    if fert < 120:
        tips.append({'icon':'↑','type':'warning','title':'Increase fertilizer application',
            'body':f'Current rate is {fert:.0f} kg/ha. Historical data shows yields above 5.5 t/ha correlate with 150–200 kg/ha.'})
    elif fert > 220:
        tips.append({'icon':'↓','type':'info','title':'Fertilizer rate is very high',
            'body':f'At {fert:.0f} kg/ha, diminishing returns may apply. Consider splitting applications.'})
    if rain < 1800:
        tips.append({'icon':'💧','type':'warning','title':'Low rainfall — supplement irrigation',
            'body':f'Rainfall of {rain:.0f} mm is below the productive range (1800–2700 mm).'})
    elif rain > 2600:
        tips.append({'icon':'⚠','type':'info','title':'High rainfall — watch for flooding',
            'body':f'At {rain:.0f} mm, excess water may cause nutrient leaching.'})
    if moist < 50:
        tips.append({'icon':'🌱','type':'warning','title':'Improve soil moisture retention',
            'body':f'Soil moisture at {moist:.0f}% is low. Apply organic mulch or compost.'})
    elif moist > 75:
        tips.append({'icon':'✓','type':'success','title':'Soil moisture is optimal',
            'body':f'At {moist:.0f}%, moisture is in the high-productive range.'})
    if temp > 29:
        tips.append({'icon':'🌡','type':'warning','title':'High temperature stress risk',
            'body':f'Temperature of {temp:.1f}°C is above optimal (25–28°C). Use heat-tolerant varieties like NSIC Rc 222.'})
    elif temp < 25:
        tips.append({'icon':'🌡','type':'info','title':'Low temperature may slow growth',
            'body':f'At {temp:.1f}°C, growth rate may be reduced. Use cold-tolerant varieties.'})

    disease_tips = [
        (0,'🌿','success','No disease detected','Excellent disease-free conditions. Continue with preventive measures and regular monitoring.'),
        (1,'⚠️','info','Very low disease pressure','Minimal disease presence. Monitor for early symptoms.'),
        (2,'🔍','warning','Low disease pressure','Early disease detected. Consider preventive fungicide application.'),
        (3,'🦠','warning','Moderate disease pressure','Significant disease presence. Apply targeted fungicides.'),
        (4,'🚨','danger','High disease pressure','Severe disease outbreak. Immediate intervention required.'),
        (5,'☠️','danger','Very high disease pressure','Critical disease situation. Emergency measures needed.'),
    ]
    for sev, icon, typ, title, body in disease_tips:
        if disease == sev:
            tips.append({'icon':icon,'type':typ,'title':title,'body':body})
    if disease >= 3:
        tips.append({'icon':'💊','type':'warning','title':'Disease management required',
            'body':f'Disease severity {disease}/5 requires immediate attention. Consult DA-CALABARZON.'})
    if category == 'low':
        tips.append({'icon':'📋','type':'danger','title':'Priority action: soil testing recommended',
            'body':'Low predicted yield suggests multiple limiting factors. Contact DA-CALABARZON extension services.'})
    return tips


# ── Auth routes ───────────────────────────────────────────────────────────────

@app.route('/login')
def login_page():
    if 'user' in session:
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/api/login', methods=['POST'])
def do_login():
    body = request.get_json(force=True)
    username = body.get('username', '').strip()
    password = body.get('password', '').strip()
    user = USERS.get(username)
    if not user or user['password'] != password:
        return jsonify({'error': 'Invalid username or password'}), 401
    session['user']     = username
    session['role']     = user['role']
    session['name']     = user['name']
    return jsonify({'role': user['role'], 'name': user['name']})

@app.route('/api/logout', methods=['POST'])
def do_logout():
    session.clear()
    return jsonify({'ok': True})

@app.route('/api/me')
def me():
    if 'user' not in session:
        return jsonify({'logged_in': False})
    return jsonify({'logged_in': True, 'user': session['user'],
                    'role': session['role'], 'name': session['name']})


# ── Admin routes ──────────────────────────────────────────────────────────────

@app.route('/api/admin/rice-areas', methods=['GET'])
@admin_required
def get_rice_areas():
    return jsonify(RICE_AREAS)

@app.route('/api/admin/rice-areas', methods=['POST'])
@admin_required
def update_rice_areas():
    body = request.get_json(force=True)
    updated = {}
    for prov_id in RICE_AREAS:
        if prov_id in body:
            try:
                val = int(body[prov_id])
                if val < 0:
                    return jsonify({'error': f'Rice area for {prov_id} must be positive'}), 400
                RICE_AREAS[prov_id] = val
                updated[prov_id] = val
            except (ValueError, TypeError):
                return jsonify({'error': f'Invalid value for {prov_id}'}), 400
    return jsonify({'ok': True, 'updated': updated, 'current': RICE_AREAS})


# ── Main routes ───────────────────────────────────────────────────────────────

@app.route('/')
@login_required
def index():
    return render_template('index.html',
        stats=json.dumps(DATA_STATS),
        role=session.get('role'),
        username=session.get('name'),
        rice_areas=json.dumps(RICE_AREAS))

@app.route('/api/predict', methods=['POST'])
@login_required
def predict():
    body      = request.get_json(force=True)
    provinces = body.get('provinces', [])
    results   = []

    for prov in provinces:
        prov_id   = prov.get('id', '')
        rice_area = RICE_AREAS.get(prov_id, 0)
        inp = {
            'avg_temp':         float(prov.get('avg_temp',         27.5)),
            'rainfall_mm':      float(prov.get('rainfall_mm',      2100)),
            'soil_moisture':    float(prov.get('soil_moisture',    65)),
            'fertilizer_kg_ha': float(prov.get('fertilizer_kg_ha', 150)),
            'rice_area':        float(rice_area),
            'disease_severity': int(prov.get('disease_severity', 2)),
        }
        df    = pd.DataFrame([inp])
        y_val = float(model.predict(df)[0])
        cat   = classify_yield(y_val)
        tips  = generate_tips(inp, y_val, cat)
        total_yield_tons = round(y_val * rice_area, 2)

        results.append({
            'id': prov_id, 'name': prov.get('name'),
            'yield': round(y_val, 3), 'rice_area': rice_area,
            'total_yield_tons': total_yield_tons,
            'category': cat, 'inputs': inp, 'tips': tips,
        })

    return jsonify({
        'results':    results,
        'thresholds': {'low': round(YIELD_LOW, 3), 'high': round(YIELD_HIGH, 3)},
        'model':      DATA_STATS,
    })

@app.route('/api/stats')
@login_required
def stats():
    return jsonify(DATA_STATS)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
