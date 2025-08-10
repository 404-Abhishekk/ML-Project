import joblib as jb
import numpy as np
model = jb.load('redwine_model.keras')
result = model.predict(np.array([[6.9,0.685,0.0,2.5,0.105,22.0,37.0,0.9966,3.46,0.57,10.6]]))
print("Predicted Quality=", result)