# main.py
import pandas as pd
import numpy as np
from config import PIPELINE_CONFIG
from pipeline_builder import get_preprocessing_pipeline
import joblib 
from pipeline_builder import build_full_model_pipeline
from config import PIPELINE_CONFIG

data = pd.DataFrame({
    'age': [22, 38, np.nan, 35],
    'fare': [7.25, 71.28, 8.05, np.nan],
    'embarked': ['S', 'C', 'S', 'S'],
    'sex': ['male', 'female', 'female', 'male'],
    'survived': [0, 1, 0, 1]
})
pipeline = get_preprocessing_pipeline(PIPELINE_CONFIG)
processed_data = pipeline.fit_transform(data)
print("Successfully processed data shape:", processed_data.shape)
print("\nFirst row of processed data:\n", processed_data[0])

X = data.drop(columns=[PIPELINE_CONFIG["target_column"]])
y = data[PIPELINE_CONFIG["target_column"]]
full_model = build_full_model_pipeline(PIPELINE_CONFIG)
full_model.fit(X, y)
raw_input = pd.DataFrame({
    'age': [25],
    'fare': [100.0],
    'embarked': ['S'],
    'sex': ['female']
})

prediction = full_model.predict(raw_input)
probability = full_model.predict_proba(raw_input)

print(f"Prediction: {'Survived' if prediction[0] == 1 else 'Did not survive'}")
print(f"Confidence: {probability.max():.2%}")
joblib.dump(full_model, "model.joblib")