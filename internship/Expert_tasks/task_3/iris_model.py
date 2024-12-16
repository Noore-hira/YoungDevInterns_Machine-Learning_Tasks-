# train_model.py
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
df= sns.load_dataset('iris')
X = df.drop('species', axis=1)
y = df['species']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model
with open("iris_model.pkl", "wb") as f:
    pickle.dump(model, f)
