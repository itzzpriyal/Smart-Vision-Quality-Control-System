import cv2
import numpy as np
import pytesseract
from datetime import datetime
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess_image(image):
    # Normalize image
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    # Convert to grayscale
    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Apply noise reduction
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    return denoised

def extract_text(image):
    # Improve OCR accuracy with image segmentation
    config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(image, config=config)
    return text

def check_expiry_date(text):
    # Improved date extraction using regex
    import re
    date_pattern = r'\d{2}[-/]\d{2}[-/]\d{4}'
    dates = re.findall(date_pattern, text)
    if dates:
        try:
            expiry_date = datetime.strptime(dates[0], "%d-%m-%Y")
            if expiry_date > datetime.now():
                return "Product is not expired"
            else:
                return "Product is expired"
        except ValueError:
            pass
    return "Could not parse expiry date"

def count_objects(image):
    # Use Canny edge detection for improved object counting
    edges = cv2.Canny(image, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)

def extract_features(image):
    # Extract color histogram
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def train_freshness_model(images, labels):
    features = [extract_features(img) for img in images]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    model = SVC(kernel='rbf', C=1.0)
    model.fit(features_scaled, labels)
    joblib.dump(model, 'freshness_model.pkl')
    joblib.dump(scaler, 'feature_scaler.pkl')

def assess_freshness(image):
    features = extract_features(image)
    scaler = joblib.load('feature_scaler.pkl')
    model = joblib.load('freshness_model.pkl')
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    return "Fresh" if prediction[0] == 1 else "Not fresh"

def main():
    # Load an image (replace with your image path)
    image = cv2.imread('product_image.jpg')
    
    # Preprocess the image
    preprocessed = preprocess_image(image)
    
    # Extract text using OCR
    text = extract_text(preprocessed)
    print("Extracted Text:", text)
    
    # Check expiry date
    expiry_status = check_expiry_date(text)
    print("Expiry Status:", expiry_status)
    
    # Count objects
    object_count = count_objects(preprocessed)
    print("Object Count:", object_count)
    
    # Assess freshness
    freshness = assess_freshness(image)
    print("Freshness:", freshness)

    # Example of how to train the freshness model (uncomment to use)
    # train_images = [cv2.imread(f'training_image_{i}.jpg') for i in range(100)]
    # train_labels = [1 if i < 50 else 0 for i in range(100)]  # 1 for fresh, 0 for not fresh
    # train_freshness_model(train_images, train_labels)

if _name_ == "_main_":
    main()
    