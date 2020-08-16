import cv2
import pickle
import numpy as np
from dataset import get_data
from feature import get_features
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

class Model:

    def __init__(self, X):
        self.classifier = SVC()
        self.scaler = StandardScaler().fit(X)

    def train(self, X_train, y_train):
        X_train = self.scaler.transform(X_train)
        self.classifier.fit(X_train, y_train)

    def test(self, X_test, y_test):
        # Accuracy
        X_test = self.scaler.transform(X_test)
        print(f"Model Accuracy: {self.classifier.score(X_test, y_test)}")
        # Confusion Matrix
        y_pred = self.classifier.predict(X_test)
        print("Model Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    def predict(self, x):
        X = self.scaler.transform([x])
        return int(self.classifier.predict(X)[0])

def get_model(filename):
    model = pickle.load(open("model.pickle", "rb"))
    return model

def main(filename):
    # Load data
    print(f"Loading data from data.h5 ...")
    vehicles, background = get_data("data.h5")

    # Extract HOG features
    print("Extracting features ...")
    vehicles_features = np.array([get_features(image).ravel() for image in vehicles])
    background_features = np.array([get_features(image).ravel() for image in background])
    X = np.vstack((vehicles_features, background_features))
    y = np.hstack((np.ones(len(vehicles_features)), np.zeros(len(background_features))))

    print(f"Input dimension: {X.shape}")
    print(f"Label dimension: {y.shape}")

    # Create Model
    model = Model(X)

    # Create training and testing data
    seed = 42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # Train Model
    print("Training model ...")
    model.train(X_train, y_train)

    # Test Model
    model.test(X_test, y_test)

    # Save Model
    print(f"Saving model to {filename} ...")
    pickle.dump(model, open(filename, "wb"))

    print("DONE!")

if __name__ == '__main__':
    main("model.pickle")
