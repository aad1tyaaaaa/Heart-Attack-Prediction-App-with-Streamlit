import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
data_path = 'heart+disease/processed.cleveland.data'
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
df = pd.read_csv(data_path, header=None, names=columns, na_values='?')

# Handle missing values
df = df.fillna(df.median())

# Convert target to binary: 0 = no disease, 1 = disease
df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

# Features and target
X = df.drop(['num', 'target'], axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Save model
with open('model.sav', 'wb') as f:
    pickle.dump(model, f)

print('Model saved as model.sav')
