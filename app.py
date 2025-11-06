import pickle
import pandas as pd

# Load the trained model
with open('NB_model.pkl', 'rb') as f:
    model = pickle.load(f)

def predict_heart_disease(data):
    """
    Predicts the likelihood of heart disease based on input data.

    Args:
        data (dict): A dictionary containing the input features.
                     Keys should match the column names used for training:
                     "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                     "thalach", "exang", "oldpeak", "slope", "ca", "thal"

    Returns:
        numpy.ndarray: An array containing the predicted class label (0-4).
    """
    # Convert input data to a pandas DataFrame
    input_df = pd.DataFrame([data])

    # Make prediction
    prediction = model.predict(input_df)

    return prediction

if __name__ == '__main__':
    # Example usage:
    # Replace with actual data for prediction
    new_patient_data = {
        "age": 63.0,
        "sex": 1.0,
        "cp": 1.0,
        "trestbps": 145.0,
        "chol": 233.0,
        "fbs": 1.0,
        "restecg": 2.0,
        "thalach": 150.0,
        "exang": 0.0,
        "oldpeak": 2.3,
        "slope": 3.0,
        "ca": 0.0,
        "thal": 6.0
    }

    prediction = predict_heart_disease(new_patient_data)
    print(f"Prediction for the new patient: {prediction[0]}")
