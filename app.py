from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

model = joblib.load("crop_yield_model.pkl")

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

data = pd.read_csv("crop_yield.csv")
data.columns = data.columns.str.strip()

# Target column
target = "Yield"

# Input features
X = data.drop(columns=[target])

# --------------------------------------------------
# IDENTIFY FEATURE TYPES
# --------------------------------------------------

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():

    return render_template(
        "index.html",
        categorical_features=categorical_features,
        numerical_features=numerical_features
    )


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    input_data = {}

    # Get values submitted from website
    for column in X.columns:

        value = request.form.get(column)

        if column in numerical_features:
            input_data[column] = float(value)

        else:
            input_data[column] = value

    # Convert input into DataFrame
    new_data = pd.DataFrame([input_data])

    # Make prediction
    prediction = model.predict(new_data)[0]

    # Return result to website
    return render_template(
        "index.html",
        prediction=prediction,
        categorical_features=categorical_features,
        numerical_features=numerical_features
    )


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)