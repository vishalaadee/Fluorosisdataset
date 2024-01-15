import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt

loaded_model, loaded_scaler = joblib.load('gradient_boosting_model_with_scaling.pkl')

user_input = input("Enter the path of the image you want to predict: ")
user_image = cv2.imread(user_input, cv2.IMREAD_GRAYSCALE)

if user_image is None:
    print("Error: Unable to load the image.")
else:
    user_image_resized = cv2.resize(user_image, (100, 100))

    user_features = user_image_resized.flatten()
    user_features_scaled = loaded_scaler.transform(user_features.reshape(1, -1))

    user_prediction = loaded_model.predict(user_features_scaled)

    plt.imshow(user_image_resized, cmap='gray')
    plt.title(f"Predicted: {'Patient' if user_prediction == 1 else 'Normal'}")
    plt.show()
