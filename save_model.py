import pickle
import pandas as pd
from predict_philanthropy import rf, X

# Save the model and feature columns
model_bundle = {
    'model': rf,
    'feature_columns': list(X.columns)
}

with open('philanthropy_model.pkl', 'wb') as f:
    pickle.dump(model_bundle, f)

print("Model and feature columns saved as 'philanthropy_model.pkl'") 