import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load preprocessed train/test data
train_df = pd.read_csv('data/processed_train.csv')
test_df = pd.read_csv('data/processed_test.csv')

# Split into X and y
X_train = train_df.drop('fake', axis=1)
y_train = train_df['fake']
X_test = test_df.drop('fake', axis=1)
y_test = test_df['fake']

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict
y_pred = rf_model.predict(X_test)

# Evaluation
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(rf_model, 'model/randomforest_model.pkl')

# Save predictions
results_rf = X_test.copy()
results_rf['actual'] = y_test
results_rf['predicted'] = y_pred
results_rf.to_csv('model/predictions_rf.csv', index=False)


results = pd.DataFrame({'actual': y_test, 'predicted': y_pred})
results.to_csv('model/predictions_rf.csv', index=False)
