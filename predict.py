import pandas as pd
from keras.models import load_model

model = load_model("trained_model.h5")
X = pd.read_csv("proposed_new_product.csv").values

prediction = model.predict(X)

# Keras always returns as 2D array
prediction = prediction[0][0]

# Re-scale
prediction += 0.1159
prediction /= 0.0000036968

print("Prediction for Proposed Product: ${}".format(prediction))

