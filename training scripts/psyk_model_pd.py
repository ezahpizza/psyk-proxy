import pandas as pd

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from category_encoders import LeaveOneOutEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("/content/mental_health_dataset_encoded.csv")

data = pd.get_dummies(data=df, columns=["occupation",
                                        "days_indoors",
                                        "stress",
                                        "mental_health_history", "work_interest",
                                        "social_anxiety",
                                        "consult_history",
                                        "mood_swings",
                                        "care_options"])

leave_encoder = LeaveOneOutEncoder()
data["country"] = leave_encoder.fit_transform(data["country"], data.iloc[:, -3])

data.info()

y = data.iloc[:, -3:]
X = data.drop(data.iloc[:, -3:], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)

roc_auc_list = []
accuracy_list = []

hist_class = MultiOutputClassifier(HistGradientBoostingClassifier(), n_jobs=-1)
print("HistGradientBoostingClassifier\n")

for i in data.iloc[:,-3:].columns:
    hist_class.fit(X_train, y_train[[i]])
    y_pred = hist_class.predict(X_test)
    roc_auc = roc_auc_score(y_test[[i]], y_pred)
    accuracy = accuracy_score(y_test[[i]], y_pred)

    roc_auc_list.append(roc_auc)
    accuracy_list.append(accuracy)

    print(f'Category name: {i}')
    print(f'{i} AUC ROC score is: {roc_auc:.3f}')
    print(f"accuracy score is: {accuracy:.3f}")
    print("\n", "-" * 50)

hist_low = MultiOutputClassifier(HistGradientBoostingClassifier(), n_jobs=-1).fit(X_train, y_train[["care_options_No"]])
hist_medium = MultiOutputClassifier(HistGradientBoostingClassifier(), n_jobs=-1).fit(X_train, y_train[["care_options_Not sure"]])
hist_high = MultiOutputClassifier(HistGradientBoostingClassifier(), n_jobs=-1).fit(X_train, y_train[["care_options_Yes"]])

pred_high = hist_high.predict(X_test)
pred_low = hist_low.predict(X_test)
pred_medium = hist_medium.predict(X_test)

print(classification_report(y_test["care_options_Yes"], pred_high))