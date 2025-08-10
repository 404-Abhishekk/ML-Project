import joblib as jb
import numpy as np
model = jb.load('redwine_model.keras')
result = model.predict(np.array([[6.3,0.48,0.04,1.1,0.046,30,99,0.9928,3.24,0.36,9.6]]))
print("Predicted Quality=", result)