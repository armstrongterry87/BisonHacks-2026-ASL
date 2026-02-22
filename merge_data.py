import pandas as pd

# Load all datasets
person1 = pd.read_csv('asl_landmarks.csv')
person2 = pd.read_csv('asl_landmarks_person2.csv')

# Combine them
combined = pd.concat([person1, person2], ignore_index=True)

# Save merged data
combined.to_csv('asl_landmarks.csv', index=False)
print(f"Combined dataset: {len(combined)} samples")
print(combined['label'].value_counts())