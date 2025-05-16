import pandas as pd

# Load raw data
train_df = pd.read_csv("data/insta_train.csv")
test_df = pd.read_csv("data/insta_test.csv")

print("Train missing values:\n", train_df.isnull().sum())
print("Test missing values:\n", test_df.isnull().sum())

from sklearn.preprocessing import StandardScaler

# Separate labels and features
X_train = train_df.drop('fake', axis=1)
y_train = train_df['fake']
X_test = test_df.drop('fake', axis=1)
y_test = test_df['fake']

# Apply StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrames
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Reattach the labels
train_final = X_train_scaled_df.copy()
train_final['fake'] = y_train.values

test_final = X_test_scaled_df.copy()
test_final['fake'] = y_test.values

# Save preprocessed datasets
train_final.to_csv("data/processed_train.csv", index=False)
test_final.to_csv("data/processed_test.csv", index=False)
