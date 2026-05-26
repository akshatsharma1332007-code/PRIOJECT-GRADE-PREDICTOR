# STUDENT GRADE PREDICTOR
# This program predicts a student's grade based on their study hours and attendance percentage.
#BY AKSHAT SHARMA

#Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
print("Libraries imported successfully.") 

# Loading dataset
df = pd.read_csv("Student_data.csv")
print(f"dataset shape: {df.shape}")
print(df.head())
print(df.describe())
print("missing values:\n", df.isnull().sum())

# Data Visualization
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.scatter(df["study_hours"], df["final_grade"], color = "blue", alpha = 0.5)
plt.xlabel("study_hours")
plt.ylabel("final_grade")
plt.title("Study Hours vs Final Grade")

plt.subplot(1,3,2)
plt.scatter(df["attendance_pct"], df["final_grade"], color = "green", alpha = 0.5)
plt.xlabel("attendance_pct")
plt.ylabel("final_grade")
plt.title("Attendance Percentage vs Final Grade")

plt.subplot(1,3,3)
plt.scatter(df["prev_score"], df["final_grade"], color = "orange", alpha = 0.5)
plt.xlabel("prev_score")
plt.ylabel("final_grade")
plt.title("Previous Score vs Final Grade")

plt.tight_layout()
plt.savefig("EDA_plots.png")
plt.show()

#corelation heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot = True, cmap= "coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")

#prepare the data
x = df[["study_hours", "attendance_pct", "prev_score", "sleep_hours"]]
y = df[["final_grade"]]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state= 42)
print(f"{len(x_train)} training samples and {len(x_test)} testing samples.")

# Train the models
lr = LinearRegression()
lr.fit(x_train, y_train)
lr_pred = lr.predict(x_test)
print("\n Trained linear regression model.")
print(f"MAE: {mean_squared_error(y_test, lr_pred):.2f}, R2: {r2_score(y_test, lr_pred):.2f}")

#Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)
rf_pred = rf.predict(x_test)
print("\n Trained random forest regressor model.")
print(f"MAE: {mean_squared_error(y_test, rf_pred):.2f}, R2: {r2_score(y_test, rf_pred):.2f}")

#Feature importance chart
features = ["study_hours", "attendance_pct", "prev_score", "sleep_hours"]
importances = rf.feature_importances_
plt.figure(figsize=(6,4))
plt.barh(features, importances, color=['#3498db','#2ecc71','#e74c3c','#f39c12'])
plt.xlabel("Feature Importance")
plt.title("Which factors most influence the final grade?")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show() 

# Predict a new student's grade
print("\n Predicting a new student's grade...")
new_student = pd.DataFrame({"study_hours": [8], "attendance_pct": [80], "prev_score": [60], "sleep_hours": [5]})
grade = rf.predict(new_student)[0]
print(f"predicted grade: {grade:.2f}/100")