import joblib
import numpy as np

# Load trained model
model = joblib.load("model/model.joblib")

# Encode product type exactly as used in training
PRODUCT_MAP = {
    "L": 0,
    "M": 1,
    "H": 2
}

def predict_failure(features_dict):
    """
    features_dict = {
        air_temp,
        process_temp,
        rpm,
        torque,
        tool_wear,
        product_type
    }
    """

    product_encoded = PRODUCT_MAP[features_dict["product_type"]]

    X = np.array([[
        features_dict["air_temp"],
        features_dict["process_temp"],
        features_dict["rpm"],
        features_dict["torque"],
        features_dict["tool_wear"],
        product_encoded
    ]])

    prob = model.predict_proba(X)[0][1]
    return prob
