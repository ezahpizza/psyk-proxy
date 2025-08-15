import polars as pl
import joblib

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from category_encoders import LeaveOneOutEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

import warnings
warnings.filterwarnings("ignore")

# Load data with polars
df = pl.read_csv("/content/mental_health_dataset_encoded.csv")

# One-hot encode categorical columns
data = df.to_dummies(
        columns=["occupation", "days_indoors", "stress", "mental_health_history", 
                "work_interest", "social_anxiety", "consult_history", 
                "mood_swings", "care_options"]
)

# Convert back to pandas for LeaveOneOutEncoder (as it doesn't support polars)
data_pandas = data.to_pandas()

# Apply Leave-One-Out encoding to country column
leave_encoder = LeaveOneOutEncoder()
data_pandas["country"] = leave_encoder.fit_transform(
    data_pandas["country"], 
    data_pandas.iloc[:, -3]
)

# Convert back to polars
data = pl.from_pandas(data_pandas)

print("Data info:")
print(f"Shape: {data.shape}")
print(f"Columns: {data.columns}")
print(data.dtypes)

# Split features and targets using polars
# Get the last 3 columns as targets
target_cols = data.columns[-3:]
y = data.select(target_cols).to_pandas()  # Convert to pandas for sklearn
X = data.select([col for col in data.columns if col not in target_cols]).to_pandas()

print(f"\nTarget columns: {target_cols}")
print(f"Features shape: {X.shape}")
print(f"Targets shape: {y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=True, random_state=42
)

# Initialize tracking lists
roc_auc_list = []
accuracy_list = []

# Train one multi-output model for all targets
print("Training HistGradientBoostingClassifier for all targets\n")

hist_class = MultiOutputClassifier(HistGradientBoostingClassifier(), n_jobs=-1)
hist_class.fit(X_train, y_train)

# Make predictions for all targets at once
y_pred = hist_class.predict(X_test)

# Calculate metrics for each target
for i, target in enumerate(target_cols):
    roc_auc = roc_auc_score(y_test.iloc[:, i], y_pred[:, i])
    accuracy = accuracy_score(y_test.iloc[:, i], y_pred[:, i])

    roc_auc_list.append(roc_auc)
    accuracy_list.append(accuracy)

    print(f'Category name: {target}')
    print(f'{target} AUC ROC score is: {roc_auc:.3f}')
    print(f"accuracy score is: {accuracy:.3f}")
    print("\n", "-" * 50)

# Save the main model
print("\nSaving model...")

# Save the main model
joblib.dump(hist_class, "model.pkl")
print("Saved main model: model.pkl")

# Save the encoder
joblib.dump(leave_encoder, "encoder.pkl")
print("Saved LeaveOneOutEncoder: encoder.pkl")

# Save feature names for future use
joblib.dump(list(X.columns), "features.pkl")
print("Saved feature names: featuress.pkl")

# Save target names
joblib.dump(target_cols, "targets.pkl")
print("Saved target names: targets.pkl")

# Create a complete model package
model_package = {
    'model': hist_class,
    'encoder': leave_encoder,
    'feature_names': list(X.columns),
    'target_names': target_cols,
    'performance': {
        'roc_auc_scores': dict(zip(target_cols, roc_auc_list)),
        'accuracy_scores': dict(zip(target_cols, accuracy_list))
    }
}

# Save complete model package
joblib.dump(model_package, "model_package.pkl")
print("Saved complete model package: model_package.pkl")

print(f"\nModel training and saving completed!")
print(f"Model performance summary:")
for i, target in enumerate(target_cols):
    print(f"  {target}: AUC = {roc_auc_list[i]:.3f}, Accuracy = {accuracy_list[i]:.3f}")

# Print classification report for the last target
if len(target_cols) > 0:
    last_target_idx = -1
    print(f"\nClassification Report for {target_cols[last_target_idx]}:")
    print(classification_report(y_test.iloc[:, last_target_idx], y_pred[:, last_target_idx]))