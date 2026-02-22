import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load ASL landmarks data
df = pd.read_csv('asl_landmarks_person1.csv')

# Separate features (X) and labels (y)
X = df.drop('label', axis=1)  # All columns except 'label'
y = df['label']  # The label column

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model
with open('asl_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\nModel saved as 'asl_model.pkl'")