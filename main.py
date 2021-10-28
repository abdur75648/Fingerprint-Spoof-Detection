import os
import cv2
import numpy as np
from tqdm import tqdm

from skimage import feature

from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

def main():
    train_x = []
    train_y = []

    test_x = []
    test_y = []

    print("Extracting LBPH (Local Binary Pattern Histogram) features from the training images (live)")
    images_live = sorted(os.listdir("data/train_live"))
    for file in tqdm(images_live):
        file = "data/train_live/"+str(file)
        
        image = cv2.imread(file)

        cv2.resize(image,(64,64))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Using Local Binary Pattern (LBP) for texture classification
        lbp = feature.local_binary_pattern(image, 24 ,8, method="uniform")

        # Getting histogram of LBP
        hist, bins = np.histogram(lbp.ravel(), bins=np.arange(0,27), range=(0,10))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        
        # Adding Data Point TO Array
        train_y.append(1) # Live Samples Are The Positive Ones
        train_x.append(hist)
    
    print("Extracting LBPH (Local Binary Pattern Histogram) features from the training images (spoof)")
    images_spoof = sorted(os.listdir("data/train_spoof"))
    for file in tqdm(images_spoof):
        file = "data/train_spoof/"+str(file)
        
        image = cv2.imread(file)

        cv2.resize(image,(64,64))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Using Local Binary Pattern (LBP) for texture classification
        lbp = feature.local_binary_pattern(image, 24 ,8, method="uniform")

        # Getting histogram of LBP
        hist, bins = np.histogram(lbp.ravel(), bins=np.arange(0,27), range=(0,10))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        
        # Adding Data Point TO Array
        train_y.append(0) # Spoof Samples Are The Negative Ones
        train_x.append(hist)

    train_x, train_y = shuffle(train_x, train_y)


    print("Extracting LBPH (Local Binary Pattern Histogram) features from the testing images (live)")
    images_live = sorted(os.listdir("data/test_live"))
    for file in tqdm(images_live):
        file = "data/test_live/"+str(file)
        
        image = cv2.imread(file)

        cv2.resize(image,(64,64))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Using Local Binary Pattern (LBP) for texture classification
        lbp = feature.local_binary_pattern(image, 24 ,8, method="uniform")

        # Getting histogram of LBP
        hist, bins = np.histogram(lbp.ravel(), bins=np.arange(0,27), range=(0,10))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        
        # Adding Data Point TO Array
        test_y.append(1) # Live Samples Are The Positive Ones
        test_x.append(hist)
    
    print("Extracting LBPH (Local Binary Pattern Histogram) features from the testing images (spoof)")
    images_spoof = sorted(os.listdir("data/test_spoof"))
    for file in tqdm(images_spoof):
        file = "data/test_spoof/"+str(file)
        
        image = cv2.imread(file)

        cv2.resize(image,(64,64))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Using Local Binary Pattern (LBP) for texture classification
        lbp = feature.local_binary_pattern(image, 24 ,8, method="uniform")

        # Getting histogram of LBP
        hist, bins = np.histogram(lbp.ravel(), bins=np.arange(0,27), range=(0,10))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        
        # Adding Data Point TO Array
        test_y.append(0) # Spoof Samples Are The Negative Ones
        test_x.append(hist)

    train_x, train_y = shuffle(train_x, train_y)
    test_x, test_y = shuffle(test_x, test_y)


    print("Training SVM Model..")
    # Train SVM on the data
    model = LinearSVC(C=100,max_iter=10000)
    model.fit(train_x,train_y)
    print("Training Completed!\n\nTesting Now...")
    pred = model.predict(test_x)

    con_matrix=confusion_matrix(test_y,pred) #,labels=["Live","Fake"])
    TP=con_matrix[0][0]
    FN=con_matrix[0][1]
    FP=con_matrix[1][0]
    TN=con_matrix[1][1]

    print("Precision of the SVM:", round((TP / (TP+FP)),3))
    print("Recall of the SVM:", round((TP / (TP+FN)),3))
    print("Accuracy of the SVM:", round(((TP + TN) / (TP + TN + FP + FN)),3))


if __name__ == "__main__":
    print("Fingerprint spoof detection system based on two-class SVM")
    main()