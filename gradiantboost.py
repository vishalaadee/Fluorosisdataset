import os
import cv2
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def load_and_resize_images(folder, target_size=(100, 100)):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img_resized = cv2.resize(img, target_size)

            img_flat = img_resized.flatten()

            print(f"Loaded and Resized Image: {filename}, Original Shape: {img.shape}, Resized Shape: {img_resized.shape}, Flattened Shape: {img_flat.shape}")

            images.append((filename, img_flat))

    return images

normal_images = load_and_resize_images("/Users/er.vishalmishra/Downloads/Fluorosisdataset/NormalTeeth")
patient_images = load_and_resize_images("/Users/er.vishalmishra/Downloads/Fluorosisdataset/sample data")

min_samples = min(len(normal_images), len(patient_images))

print(f"Number of normal samples: {len(normal_images)}")
print(f"Number of patient samples: {len(patient_images)}")
print(f"Minimum number of samples: {min_samples}")

normal_labels = np.zeros(min_samples)
patient_labels = np.ones(min_samples)

X = np.vstack((np.array([img[1] for img in normal_images[:min_samples]]),
               np.array([img[1] for img in patient_images[:min_samples]])))

print(f"Shape of X after stacking: {X.shape}")

y = np.concatenate((normal_labels, patient_labels))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Shape of X_train_scaled: {X_train_scaled.shape}")
print(f"Shape of X_test_scaled: {X_test_scaled.shape}")

classifier = GradientBoostingClassifier()
classifier.fit(X_train_scaled, y_train)

y_pred = classifier.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_rep)

import joblib
joblib.dump((classifier, scaler), 'gradient_boosting_model_with_scaling.pkl')

