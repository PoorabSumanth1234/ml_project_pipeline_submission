import joblib
from main import pipeline 
import pandas as pd 
joblib.dump(pipeline, 'preprocessor_pipeline.joblib')
print("Saved")
loaded_pipeline = joblib.load('preprocessor_pipeline.joblib')
new_data = pd.DataFrame({
    'age': [30],
    'fare': [25.0],
    'embarked': ['C'],
    'sex': ['female']
})

clean_new_data = loaded_pipeline.transform(new_data)
print("\nProcessed New Data Row:")
print(clean_new_data)