import warnings
warnings.filterwarnings("ignore")

from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained pipeline
model = joblib.load("models/best_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]

    if file.filename == "":
        return "No file selected"

    # Read CSV
    df = pd.read_csv(file)

    # ----------------------------
    # Feature Engineering
    # ----------------------------

    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])

    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day"] = df["trans_date_trans_time"].dt.day
    df["month"] = df["trans_date_trans_time"].dt.month

    df["dob"] = pd.to_datetime(df["dob"])

    df["age"] = 2026 - df["dob"].dt.year

    # ----------------------------
    # Drop columns
    # ----------------------------

    drop_columns = [

        "Unnamed: 0",

        "trans_date_trans_time",

        "cc_num",

        "first",

        "last",

        "street",

        "trans_num",

        "dob"

    ]

    df.drop(columns=drop_columns, inplace=True)

    # ----------------------------
    # Remove target if uploaded
    # ----------------------------

    if "is_fraud" in df.columns:
        df = df.drop("is_fraud", axis=1)

    # ----------------------------
    # Prediction
    # ----------------------------

    prediction = model.predict(df)

    fraud = int(sum(prediction))

    legitimate = len(prediction) - fraud

    fraud_percentage = round((fraud / len(prediction)) * 100, 2)

    return render_template(

        "index.html",

        total=len(prediction),

        fraud=fraud,

        legitimate=legitimate,

        percentage=fraud_percentage

    )


if __name__ == "__main__":
    app.run(debug=True)