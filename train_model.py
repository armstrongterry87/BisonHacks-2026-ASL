import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import numpy as np

# Load data
df = pd.read_csv('asl_landmarks.csv')

print(f"Total samples: {len(df)}")
print(f"Samples per letter:\n{df['label'].value_counts()}")

X = df.drop('label', axis=1).values
y = df['label']

# Normalize each sample (MUST MATCH TRANSLATOR)
X_normalized = []
for row in X:
    landmarks = row.reshape(21, 3)
    wrist = landmarks[0]
    centered = landmarks - wrist
    scale = np.linalg.norm(centered[12])
    if scale > 0:
        normalized = centered / scale
    else:
        normalized = centered
    X_normalized.append(normalized.flatten())

X_normalized = np.array(X_normalized)

# Split and train
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=300, max_depth=25, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save
with open('asl_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nModel saved as 'asl_model.pkl'")