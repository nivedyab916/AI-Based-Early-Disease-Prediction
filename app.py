from flask import Flask, render_template, request
import joblib
import sqlite3
from datetime import datetime

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


# ---------------- DATABASE ----------------
def init_db():
    conn = sqlite3.connect("database/disease.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            prediction TEXT,
            probability REAL,
            risk TEXT,
            date TEXT
        )
    """)

    conn.commit()
    conn.close()


init_db()


# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------- PREDICTION ----------------
@app.route("/predict", methods=["POST"])
def predict():

    name = request.form["name"]

    features = [[
        float(request.form["pregnancies"]),
        float(request.form["glucose"]),
        float(request.form["bloodpressure"]),
        float(request.form["skinthickness"]),
        float(request.form["insulin"]),
        float(request.form["bmi"]),
        float(request.form["dpf"]),
        float(request.form["age"])
    ]]

    # Scale data
    scaled = scaler.transform(features)

    # Prediction
    prediction = model.predict(scaled)[0]

    # Probability
    probability = model.predict_proba(scaled)[0][1] * 100

    if prediction == 1:
        result = "Diabetic"
    else:
        result = "Non-Diabetic"

    # Risk Level
    if probability < 30:
        risk = "Low"
    elif probability < 70:
        risk = "Medium"
    else:
        risk = "High"

    # Recommendations
    if prediction == 1:
        recommendation = [
            "Consult a physician.",
            "Reduce sugar intake.",
            "Exercise regularly.",
            "Monitor blood glucose levels."
        ]
    else:
        recommendation = [
            "Maintain a healthy lifestyle.",
            "Exercise regularly.",
            "Eat a balanced diet.",
            "Get regular health check-ups."
        ]

    # Save to database
    conn = sqlite3.connect("database/disease.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO history(name, prediction, probability, risk, date)
        VALUES (?, ?, ?, ?, ?)
    """, (
        name,
        result,
        round(probability, 2),
        risk,
        datetime.now().strftime("%d-%m-%Y %H:%M")
    ))

    conn.commit()
    conn.close()

    return render_template(
        "result.html",
        name=name,
        prediction=result,
        probability=round(probability, 2),
        risk=risk,
        recommendation=recommendation
    )


# ---------------- HISTORY ----------------
@app.route("/history")
def history():

    conn = sqlite3.connect("database/disease.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT name, prediction, probability, risk, date
        FROM history
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()

    conn.close()

    return render_template("history.html", rows=rows)


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)