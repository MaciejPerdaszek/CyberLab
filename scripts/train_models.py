import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

PROCESSED_PATH = '../data/processed/'
MODELS_PATH = '../models/'
RESULTS_PATH = '../results/'

os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH + 'confusion_matrices', exist_ok=True)

X = pd.read_csv(PROCESSED_PATH + 'X_scaled.csv').values
y = pd.read_csv(PROCESSED_PATH + 'y.csv').values.ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42)
}

results = []

for name, model in models.items():
    print(f"Trening modelu: {name}")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    results.append({'model': name, **report['accuracy']})

    joblib.dump(model, MODELS_PATH + f"{name}.save")

    cm = confusion_matrix(y_test, y_pred)
    pd.DataFrame(cm).to_csv(RESULTS_PATH + f'confusion_matrices/cm_{name}.csv', index=False)

    print(classification_report(y_test, y_pred))

pd.DataFrame(results).to_csv(RESULTS_PATH + 'metrics.csv', index=False)
print("Trening i ewaluacja zakończone.")