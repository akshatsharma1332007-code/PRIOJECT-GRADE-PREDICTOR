# STUDENT GRADE PREDICTOR
# Predicts a student's final grade (G3, scale 0-20) from study habits, past grades and lifestyle.
# Dataset: UCI Student Performance (Mathematics course, 395 students)
# BY AKSHAT SHARMA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------- Load and clean ----------
import os
df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "student-mat.csv"), sep=None, engine="python")
print(f"dataset shape: {df.shape}")
print("missing values:", df.isnull().sum().sum())
print("duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()

# ---------- Feature engineering ----------
df["avg_prev_grade"] = (df["G1"] + df["G2"]) / 2      # average of earlier period grades
df["alcohol_total"] = df["Dalc"] + df["Walc"]         # weekday + weekend alcohol use

features = ["studytime", "failures", "absences", "health",
            "freetime", "goout", "alcohol_total", "avg_prev_grade"]
target = "G3"

# ---------- EDA ----------
plt.figure(figsize=(12, 4))
for i, col in enumerate(["studytime", "absences", "avg_prev_grade"], start=1):
    plt.subplot(1, 3, i)
    plt.scatter(df[col], df[target], alpha=0.4)
    plt.xlabel(col)
    plt.ylabel("final grade (G3)")
    plt.title(f"{col} vs G3")
plt.tight_layout()
plt.savefig("EDA_plots.png")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df[features + [target]].corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

# ---------- Prepare data ----------
x = df[features]
y = df[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(f"{len(x_train)} training samples and {len(x_test)} testing samples.")

def evaluate(name, model, pred):
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    cv = cross_val_score(model, x, y, cv=5, scoring="r2")
    print(f"{name} | RMSE: {rmse:.2f} | MAE: {mae:.2f} | R2: {r2:.2f} | 5-fold CV R2: {cv.mean():.2f} (+/- {cv.std():.2f})")

# ---------- Models ----------
lr = LinearRegression()
lr.fit(x_train, y_train)
evaluate("Linear Regression", lr, lr.predict(x_test))

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)
evaluate("Random Forest", rf, rf.predict(x_test))

# ---------- Feature importance ----------
imp = pd.Series(rf.feature_importances_, index=features).sort_values()
plt.figure(figsize=(7, 4))
imp.plot(kind="barh", color="#3498db")
plt.xlabel("Feature Importance")
plt.title("Which factors most influence the final grade?")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()
