from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("crop_yield_model.pkl")

data = pd.read_csv("crop_yield.csv")
data.columns = data.columns.str.strip()

target = "Yield"
X = data.drop(columns=[target])

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

# Get unique values for categorical dropdowns
categorical_options = {}

for column in categorical_features:
    categorical_options[column] = sorted(
        data[column].dropna().astype(str).unique().tolist()
    )


@app.route("/")
def home():
    return render_template(
        "index.html",
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        categorical_options=categorical_options
    )


@app.route("/predict", methods=["POST"])
def predict():
    input_data = {}

    for column in X.columns:
        value = request.form.get(column)

        if column in numerical_features:
            input_data[column] = float(value)
        else:
            input_data[column] = value

    new_data = pd.DataFrame([input_data])

    prediction = model.predict(new_data)[0]

    return render_template(
        "index.html",
        prediction=prediction,
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        categorical_options=categorical_options
    )


if __name__ == "__main__":
    app.run(debug=True)