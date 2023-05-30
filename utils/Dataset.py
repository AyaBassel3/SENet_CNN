import cv2
import numpy as np
import os
from numpy import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

from tqdm.auto import tqdm
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure



from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import interpolation
def moments(image):
    c0,c1 = np.mgrid[:image.shape[0],:image.shape[1]] # create a mesh grid
    totalImage = np.sum(image) #sum of pixels
    m0 = np.sum(c0*image)/totalImage #mu_x
    m1 = np.sum(c1*image)/totalImage #mu_y
    m00 = np.sum((c0-m0)**2*image)/totalImage #var(x)
    m11 = np.sum((c1-m1)**2*image)/totalImage #var(y)
    m01 = np.sum((c0-m0)*(c1-m1)*image)/totalImage #covariance(x,y)
    mu_vector = np.array([m0,m1]) # these are \mu_x, \mu_y respectively
    covariance_matrix = np.array([[m00,m01],[m01,m11]]) # Do you see a similarity between the covariance matrix
    return mu_vector, covariance_matrix

def deskew(image):
    c,v = moments(image)
    alpha = v[0,1]/v[0,0]
    affine = np.array([[1,0],[alpha,1]])
    ocenter = np.array(image.shape)/2.0
    offset = c-np.dot(affine,ocenter)
    return interpolation.affine_transform(image,affine,offset=offset)


def read_dataset(data_type,dim):
    img_count=0
    data_set=[]
    labels=[]
    root=os.path.join('D:/Masters/TU Dortmund/3rd Semester/group project/algo/data/', data_type)






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
                res=cv2.resize(deskew(gray), dsize=(dim, dim), interpolation=cv2.INTER_CUBIC)
                fd, res_image = hog(res, orientations=9, pixels_per_cell=(8, 8),
                                    cells_per_block=(2, 2), visualize=True, multichannel=False)

                img_count=img_count+1
                lbl_out = np.zeros(15)
                #lbl_out[int(symbol)] = 1
                print(img_count)
                data_set.append(res_image/np.max(res_image))
                labels.append(symbol)

    data=np.array(data_set) #np array ashan feeh features ktiir
    labels=np.array(labels)
    np.save('data_set',data)
    np.save('labels',labels)

    return data,labels

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