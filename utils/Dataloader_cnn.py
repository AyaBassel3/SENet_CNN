import cv2
import numpy as np
import os
from numpy import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np







def read_dataset(data_type,dim):
    img_count=0
    data_set=[]
    labels=[]
    root=os.path.join('D:/Masters/TU Dortmund/3rd Semester/group project/pythonProject/data/', data_type)
    classes=np.array(os.listdir(root))
    num_classes = len(classes)
    class_dict = {classes[i]: i for i in range(num_classes)}






    for symbol in os.listdir(root):
            symbol_dir=os.path.join(root,symbol)
            if os.path.isfile(symbol_dir):
                continue
            for img in os.listdir(symbol_dir):
                img_path=os.path.join(symbol_dir, img)
                img=cv2.imread(img_path)
                gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                #thersh,bin= cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
                #kernel_aya = np.ones((5, 5), np.float32) / 25
                #dst = (255-(cv2.filter2D(gray, -1, gray)))
                #canny_output = cv2.Canny(gray, 20, 200)
                #bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
                #res=cv2.resize(deskew(gray), dsize=(dim, dim), interpolation=cv2.INTER_CUBIC)
                #fd, res_image = hog(res, orientations=9, pixels_per_cell=(8, 8),
                #                    cells_per_block=(2, 2), visualize=True, multichannel=False)

                res = cv2.resize(gray, dsize=(dim, dim), interpolation=cv2.INTER_CUBIC)
                img_count = img_count + 1
                label= class_dict[symbol]
                lbl_out = np.zeros(15)
                lbl_out[int(label)] = 1
                print(img_count)
                data_set.append(res / np.max(res))
                labels.append(lbl_out) #hot encoded

    data = np.array(data_set)  # np array ashan feeh features ktiir
    labels = np.array(labels)
    np.save('data_set', data)
    np.save('labels', labels)
    return data, labels

def plot25(train_data):
    indexes = np.random.randint(0, train_data.shape[0], size=25)
    images = train_data[indexes]

    # plot 25 random digits
    plt.figure(figsize=(5, 5))
    for i in range(len(indexes)):
        plt.subplot(5, 5, i + 1)
        image = images[i]
        plt.imshow(image, cmap='gray')
        plt.axis('off')

def flatten(data):
    return np.reshape(data,[-1,data.shape[1]*data.shape[2]]) #1d array

def PCAmethod(train_data,test_data,n):
    train_data=flatten(train_data)
    test_data=flatten(test_data)
    pca = PCA(n_components=n)
    return pca.fit_transform(train_data) , pca.transform(test_data)