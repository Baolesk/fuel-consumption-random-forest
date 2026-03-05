import joblib
import pandas as pd

model = joblib.load("C:/Users/baole/Documents/Fuel_Consumption_RandomForest/models/random_forest_l100km.pkl")

rpm = 2200
hp = 15

data = pd.DataFrame([[rpm, hp]],
                    columns=["Engine RPM(rpm)",
                             "Horsepower (At the wheels)(hp)"])

prediction = model.predict(data)

print("Predicted fuel consumption:", prediction[0], "L/100km")