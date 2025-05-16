import pandas as pd

# Load preprocessed data
train_df = pd.read_csv('data/processed_train.csv')
test_df = pd.read_csv('data/processed_test.csv')

X_train = train_df.drop('fake', axis=1)
y_train = train_df['fake']

X_test = test_df.drop('fake', axis=1)
y_test = test_df['fake']
 
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')  # Avoids warning

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# Confusion Matrix and Classification Report
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

import joblib
joblib.dump(model, 'model/xgboost_model.pkl')

# Save predictions
results = X_test.copy()
results['actual'] = y_test
results['predicted'] = y_pred
results.to_csv('model/predictions.csv', index=False)

results = pd.DataFrame({'actual': y_test, 'predicted': y_pred})
results.to_csv('model/predictions_xgb.csv', index=False)



