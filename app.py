from flask import Flask, render_template, request, jsonify
from src.pipeline.prediction_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

@app.route("/")
def home_page():
    return render_template("index.html")
@app.route("/predict",methods = ["GET","POST"])
def predict():
    if request.method == "GET":
        return render_template("form.html")
    else:
        data = CustomData(
            tuoi = int(request.form.get("age")),
            c_cao= float(request.form.get("height_cm")),
            c_nang =float(request.form.get("weight_kg")),
            nhip_tim = float(request.form.get("heart_rate")),
            huyet_ap = float(request.form.get("blood_pressure")),
            gio_ngu = int(request.form.get("sleep_hours")),
            dinh_duong = int(request.form.get("nutrition_quality")),
            hd = int(request.form.get("activity_index")),
            gt =request.form.get("gender"),
            hutthuoc = request.form.get("smokes")
        )
        final_data = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_data)
        result = round(pred[0],2)
        return render_template("result.html",final_result =result)


if __name__ =="__main__":
    app.run(host = "0.0.0.0",port = 8080)