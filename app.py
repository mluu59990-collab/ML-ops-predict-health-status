from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from pymongo import MongoClient
from src.pipeline.prediction_pipeline import PredictPipeline, CustomData, get_available_versions
from src.auth.routes import auth_bp, login_required
from src.auth.models import HistoryModel
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

# ── Cau hinh ─────────────────────────────────────────────────────────────────
app.secret_key = os.environ.get("SECRET_KEY", "fitness-secret-key-change-in-prod")

# ── Ket noi MongoDB ───────────────────────────────────────────────────────────
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME   = os.environ.get("MONGO_DB_NAME", "fitness_db")

client = MongoClient(MONGO_URI)
db     = client[DB_NAME]
app.extensions["mongo_db"] = db

# ── Blueprint ─────────────────────────────────────────────────────────────────
app.register_blueprint(auth_bp)


# ── Inject user vao moi template ─────────────────────────────────────────────
@app.context_processor
def inject_user():
    return {
        "current_username": session.get("username"),
        "is_logged_in": bool(session.get("user_id")),
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def home_page():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))
    return render_template("index.html")


@app.route("/dashboard")
@login_required
def dashboard():
    history_model = HistoryModel(db)
    user_id = session["user_id"]

    stats = history_model.stats(user_id)

    recent_raw = history_model.get_by_user(user_id, limit=5)
    recent = []
    for r in recent_raw:
        r["date_str"] = r["created_at"].strftime("%d/%m/%Y %H:%M")
        recent.append(r)

    return render_template(
        "dashboard.html",
        username=session.get("username"),
        stats=stats,
        recent=recent,
    )


@app.route("/api/model-versions")
@login_required
def api_model_versions():
    """API tra ve danh sach version hien co tren MLflow Registry."""
    versions = get_available_versions()
    return jsonify(versions)


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if request.method == "GET":
        # Lay danh sach version de hien thi combobox
        model_versions = get_available_versions()
        return render_template("form.html", model_versions=model_versions)

    # Lay version duoc chon tu form
    selected_version = request.form.get("model_version") or None

    # Lay du lieu form
    input_fields = {
        "age":               int(request.form.get("age")),
        "height_cm":         float(request.form.get("height_cm")),
        "weight_kg":         float(request.form.get("weight_kg")),
        "heart_rate":        float(request.form.get("heart_rate")),
        "blood_pressure":    float(request.form.get("blood_pressure")),
        "sleep_hours":       float(request.form.get("sleep_hours")),
        "nutrition_quality": float(request.form.get("nutrition_quality")),
        "activity_index":    float(request.form.get("activity_index")),
        "gender":            request.form.get("gender"),
        "smokes":            request.form.get("smokes"),
    }

    data = CustomData(
        tuoi       = input_fields["age"],
        c_cao      = input_fields["height_cm"],
        c_nang     = input_fields["weight_kg"],
        nhip_tim   = input_fields["heart_rate"],
        huyet_ap   = input_fields["blood_pressure"],
        gio_ngu    = input_fields["sleep_hours"],
        dinh_duong = input_fields["nutrition_quality"],
        hd         = input_fields["activity_index"],
        gt         = input_fields["gender"],
        hutthuoc   = input_fields["smokes"],
    )

    final_data = data.get_data_as_dataframe()

    # Du doan voi version da chon
    pred = PredictPipeline().predict(final_data, model_version=selected_version)
    raw_result = round(pred[0], 2)

    result_label = "Fit" if raw_result == 1 else "Not Fit"

    # Luu lich su vao MongoDB (them model_version vao record)
    HistoryModel(db).save(
        user_id       = session["user_id"],
        input_data    = input_fields,
        result        = result_label,
        model_version = selected_version or "latest",
    )

    return render_template(
        "result.html",
        final_result=raw_result,
        result_label=result_label,
        model_version=selected_version or "latest",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)