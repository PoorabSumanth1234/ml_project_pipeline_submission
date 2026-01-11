PIPELINE_CONFIG = {
    "numeric_features": ["age", "fare"],
    "categorical_features": ["embarked", "sex"],
    "num_impute_strategy": "median",
    "scaling_method": "standard", 
    "target_column": "survived"
}
