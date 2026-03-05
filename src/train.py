import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import os

FEATURES = [
    "Engine RPM(rpm)",
    "Horsepower (At the wheels)(hp)"
]

TARGET = "Litres Per 100 Kilometer(Instant)(l/100km)"

df = pd.read_csv("C:/Users/baole/Documents/Fuel_Consumption_RandomForest/data/sample.csv")
df.replace("-", pd.NA, inplace=True)
df = df.apply(pd.to_numeric, errors="coerce")
df = df.dropna()
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=80,        # giảm số cây
    max_depth=12,           # giới hạn độ sâu
    min_samples_leaf=5,     # tránh cây quá chi tiết
    random_state=42,
    n_jobs=-1               # chạy nhanh hơn
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("R2 score:", r2_score(y_test, y_pred))
os.makedirs("models", exist_ok=True)
joblib.dump(model, "C:/Users/baole/Documents/Fuel_Consumption_RandomForest/models/random_forest_l100km.pkl")
print("Model saved.")