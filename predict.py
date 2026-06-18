import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Load the dataset
# Replace 'movies.csv' with the actual path to your downloaded dataset file
try:
    df = pd.read_csv("movies.csv")
    print("Dataset Loaded Successfully!")
except FileNotFoundError:
    print(
        "Error: 'movies.csv' not found. Please ensure the file is in the same directory."
    )
    exit()

# Print the original columns to verify what we are working with
print("\nOriginal Dataset Preview:")
print(df.head())

# 2. Data Preprocessing
# Fill any missing values (NaNs) to avoid errors during training
df["Runtime"] = df["Runtime"].fillna(df["Runtime"].median())
df["Votes"] = df["Votes"].fillna(df["Votes"].median())

# Handle Categorical (Text) Columns using One-Hot Encoding
# This converts columns like 'Genre', 'Director', and 'Actors' into numbers
categorical_cols = ["Genre", "Director", "Actors"]
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 3. Define Features (X) and Target (y)
# 'Rating' is what we want to predict. Everything else goes into X.
X = df_encoded.drop(columns=["Rating"])
y = df_encoded["Rating"]

# 4. Split the data into Training and Testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Initialize and Train the Model
print("\nTraining the RandomForestRegressor model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Make Predictions
y_pred = model.predict(X_test)

# 7. Evaluate the Model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n=== Model Performance Evaluation ===")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2) Score: {r2:.2f}")

# 8. Show a quick comparison of actual vs predicted ratings
comparison_df = pd.DataFrame({"Actual Rating": y_test, "Predicted Rating": y_pred})
print("\nSample Predictions:")
print(comparison_df.head())
