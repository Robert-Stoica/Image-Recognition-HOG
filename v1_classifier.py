import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

Testing_Path = r'../../data/testing'
Trainig_Path = r'../../data/training'


# get data with their labels from data files
def get_data(fdir, Testing=False, size=128):
    X = []
    Y = []
    filenames = []
    fdir = os.path.abspath(fdir)
    labels = {}
    print(fdir)
    if not Testing:
        files = os.listdir(fdir)
        counter = 0

        # print(files)
        for file in files:
            file = os.path.join(fdir, file)
            if not os.path.isdir(file):
                continue
            label = os.path.basename(file)
            labels[counter] = label
            images = os.listdir(file)
            # print(file)
            # print(images)
            for imagepath in images:
                imagepath = os.path.join(file, imagepath)
                # print(imagepath)
                try:
                    # image = im.imread(imagepath)
                    image = cv2.imread(imagepath)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    image = cv2.resize(image, (size, size))
                    X.append(image)
                    Y.append(counter)
                    filenames.append(imagepath)
                except Exception as e:
                    print(e)
                    continue
            counter += 1
    else:
        files = sorted(os.listdir(fdir), key=lambda x: int(os.path.splitext(x)[0])) # sort filenames in ascending order
        counter = 0
        for file in files:
            file = os.path.join(fdir, file)
            try:
                image = cv2.imread(file)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (size, size))
                X.append(image)
                Y.append(counter)
                filenames.append(file)
            except Exception as e:
                pass
            counter += 1
    X, Y = np.array(X, dtype=np.uint8), np.array(Y, dtype=np.int32)

    return X, Y, labels, filenames


def get_hog(X):
    hog = cv2.HOGDescriptor()
    new_X = []
    i = 0
    for x in X:
        print(f"Processing image {i}")
        i += 1
        new_X.append(hog.compute(x))
    return np.array(new_X)


def calculate_average_precision(y_true, y_pred):
    # Ensure y_true and y_pred have the same shape
    if y_true.shape != y_pred.shape:
        # If y_true has fewer elements, remove extra elements from y_pred
        if y_true.shape[0] < y_pred.shape[0]:
            y_pred = y_pred[:y_true.shape[0]]
        # If y_pred has fewer elements, remove extra elements from y_true
        else:
            y_true = y_true[:y_pred.shape[0]]

    # Calculate the number of correct predictions
    num_correct = np.sum(np.equal(y_true, y_pred))
    # Calculate the average precision
    average_precision = num_correct / len(y_pred)
    return average_precision


def main():
    # Read the training data
    X, y, labels, _ = get_data(Trainig_Path)
    print(X.shape)
    hog_X = get_hog(X)
    print(hog_X.shape)

    # Split the training data into a training and validation set
    X_train, X_test, y_train, y_test = train_test_split(hog_X, y, test_size=.2, stratify=y, random_state=0)

    # Train the classifier
    sc = svm.SVC(kernel="poly")
    sc.fit(X_train, y_train)
    score = sc.score(X_test, y_test)
    print(f"Model SVC Poly accuracy is:: {score}")

    # Predict the classes for the test data
    y_pred = sc.predict(X_test)
    # Calculate the average precision
    average_precision = calculate_average_precision(y_test, y_pred)
    print(f"Average precision: {average_precision:.4f}")

    # Read the test data
    X_test, _, _, test_filenames = get_data(Testing_Path, Testing=True)
    hog_X_test = get_hog(X_test)

    # Predict the classes for the test data
    y_test_pred = sc.predict(hog_X_test)

    # Write the predictions to an output file
    try:
        with open("run3.txt", "w") as f:
            for i in range(len(test_filenames)):
                predicted_class = labels[y_test_pred[i]]
                filename = test_filenames[i].split("\\")[-1]
                print(filename + " " + predicted_class)
                f.write(filename + " " + predicted_class + "\n")
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
