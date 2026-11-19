from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "static"),
    static_folder=str(BASE_DIR / "static"),
)

bundle = joblib.load(BASE_DIR / "machine_learning_bundle.pkl")
scaler = bundle["scaler"]
encoder = bundle["encoder"]
model = bundle["model"]

numeric_features = [
    "Age",
    "Review_Influence_Score",
    "Social_Ads_Influence_Score",
    "Price_Comparison_Frequency",
    "Discount_Importance_Score",
    "Impulse_Purchase_Frequency",
    "Online_Shopping_Satisfaction_Score",
    "Trust_In_Online_Reviews_Score",
    "Regret_After_Purchase_Frequency",
]

categorical_features = encoder.feature_names_in_.tolist()
categorical_options = {
    column: categories.tolist()
    for column, categories in zip(
        categorical_features,
        encoder.categories_,
    )
}


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not payload or "data" not in payload:
        return jsonify({"detail": "Request body must include a data object"}), 400

    values = payload["data"]
    missing = [
        name
        for name in numeric_features + categorical_features
        if name not in values
    ]
    if missing:
        return jsonify({"detail": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        row = pd.DataFrame([values])
        numeric = row[numeric_features].apply(pd.to_numeric, errors="raise")
        categorical = row[categorical_features]

        categorical_encoded = encoder.transform(categorical)
        encoded_names = encoder.get_feature_names_out(categorical_features)

        processed_row = pd.DataFrame(
            np.hstack([categorical_encoded.toarray(), numeric.to_numpy()])
            if hasattr(categorical_encoded, "toarray")
            else np.hstack([categorical_encoded, numeric.to_numpy()]),
            columns=list(encoded_names) + numeric_features,
        )
        processed = scaler.transform(
            processed_row.reindex(columns=scaler.feature_names_in_)
        )

        if processed.shape[1] != model.n_features_in_:
            return jsonify({
                "detail": (
                    "The model bundle is inconsistent: preprocessing produced "
                    f"{processed.shape[1]} features, but the model expects "
                    f"{model.n_features_in_}."
                )
            }), 400

        prediction = float(model.predict(processed)[0])
        return jsonify({
            "prediction": round(prediction, 2),
            "target": "Likely To Buy After Ad",
        })
    except Exception as exc:
        return jsonify({"detail": str(exc)}), 400


@app.get("/schema")
def schema():
    return jsonify({
        "categorical": categorical_options,
        "numeric": numeric_features,
    })


if __name__ == "__main__":
    app.run(debug=True)
