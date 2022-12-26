#--------Importing All the required Libraries ---------
 
import numpy as np
import pandas as pd                               
import os
import csv
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pickle
  

# ------------Setting Path for Images (FAll and No Fall )----------

folder1="/Fall Detection/Fall"
folder2="/Fall Detection/No fall"


#-------------- Preprocessing and Feature Extraction for Folder 1 (Fall)
i=0
for filename in os.listdir(folder1):
    #Defining the path
    path=os.path.join(folder1,filename)
    a=cv2.imread(path)
    
    #resize total  image size to 100 x 100
    resize=(100,100)
    img=cv2.resize(a,resize)
    
    #grayscaling the image dataset
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2= cv2.GaussianBlur(gray,(5,5),0)#gaussian Image
        
    # creating a Histograms Equalization for folder1(Fall)
    equ = cv2.equalizeHist(img2)
    # Canny Edge Detection 
    canny=cv2.Canny(equ,100,50)
   

    #Applying Brisk Discriptor folder1 (Fall)
    brisk = cv2.BRISK_create()
    keypoint,descriptor = brisk.detectAndCompute(canny,None)
  
    #convert the descriptor array into a dataframe format
    out=pd.DataFrame(descriptor)
    print("Fall ",i," : ", out.shape)
    i=i+1
  
#Store the data into a csv file 
    csv_data=out.to_csv('Fall1.csv', mode='a', header=False,index=False)

#---------------Reading the data from Csv File -----------------------
#reading previously saved feature descriptor csv file of folder1 and save it into a dataframe
data1 = pd.read_csv('Fall1.csv.csv',header=None,dtype='uint8')

data1=data1.astype(np.uint8) 
data1

#-------------- Preprocessing and Feature Extraction for Folder 2 (No Fall)
i=0
for filename in os.listdir(folder2):
    #Defining the path
    path=os.path.join(folder2,filename)
    a=cv2.imread(path)
    
    #resize total  image size to 100 x 100
    resize=(100,100)
    img=cv2.resize(a,resize)
    
    #grayscaling the image dataset
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2= cv2.GaussianBlur(gray,(5,5),0)#gaussian Image
        
    # creating a Histograms Equalization for folder1( no Fall)
    equ = cv2.equalizeHist(img2)
    # Canny Edge Detection 
    canny=cv2.Canny(equ,100,50)
   

    #Applying Brisk Discriptor folder2 ( NoFall)
    brisk = cv2.BRISK_create()
    keypoint,descriptor = brisk.detectAndCompute(canny,None)
  
    #convert the descriptor array into a dataframe format
    out=pd.DataFrame(descriptor)
    print("No Fall ",i," : ", out.shape)
    i=i+1
  
    #Store the data into a csv file 
    csv_data=out.to_csv('No Fall1.csv', mode='a', header=False,index=False)


#----------------Reading the CSv file for Folder 2 --------------------
#reading previously saved feature descriptor csv file of folder2 and save it into a dataframe
data2= pd.read_csv('No Fall1.csv',header=None,dtype='uint8')
data2=data2.astype(np.uint8)
data2

#-------------------append The data----------------------
data=data1.append(data2)

data
#-----------save appended data into a csv file---------
csv_data=data.to_csv('finalData.csv', mode='a', header=False,index=False)


#---------------------------------------------------
#---------Applying Kmeans----------
kmeans = KMeans(n_clusters=3)
kmeans.fit(data)

#----------save the model to disk----------------
filename = 'Kmeans.sav'
pickle.dump(kmeans, open(filename, 'wb'))

#----------calculate histogram of trained kmeans---------
hist = np.histogram(kmeans.labels_,bins=[0,1,2,3])

#-------------------------------------------------------------------

#performing kmeans prediction on the folder1 with the pretrained kmeans model

#initialising i=0; as it is the first class
i=0
data=[]
#k=0

for filename in os.listdir(folder1):
    #path
    path=os.path.join(folder1,filename)
    a=cv2.imread(path)
    
    #resize image
    resize=(100,100)
    img=cv2.resize(a,resize)
    
    #gray image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Initiate FAST detector
    star = cv2.xfeatures2d.StarDetector_create()
    # Initiate BRIEF extractor
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    # find the keypoints with STAR
    kp = star.detect(img,None)
    # compute the descriptors with BRIEF
    keypoints, descriptors = brief.compute(gray, kp)
    
    out=pd.DataFrame(descriptors)
    
    array_double = np.array(out, dtype=np.double)
    try:
        a=kmeans.predict(array_double)
    except:
        print(filename)
    hist=np.histogram(a,bins=[0,1,2,3])
    
    #append the dataframe into the array 
    data.append(hist[0])
    
    
#convert Array to Dataframe and append to the list
Output = pd.DataFrame(data)
#add row class 
Output["Class"] = i 
csv_data=Output.to_csv('finalFolder1.csv', mode='a',header=False,index=False)

#---------------------------------------------------------------------------------
#performing kmeans prediction on the folder2 with the pretrained kmeans model

#initialising i=1; as its the 2nd class
i=1
data=[]
k=0
for filename in os.listdir(folder2):
    path=os.path.join(folder2,filename)
    a=cv2.imread(path)
    
    #resize image
    resize=(100,100)
    img=cv2.resize(a,resize)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Initiate FAST detector
    star = cv2.xfeatures2d.StarDetector_create()
    # Initiate BRIEF extractor
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    # find the keypoints with STAR
    kp = star.detect(img,None)
    # compute the descriptors with BRIEF
    keypoints, descriptors = brief.compute(gray, kp)
    
    out=pd.DataFrame(descriptors)

    array_double = np.array(out, dtype=np.double)
    try:
        a=kmeans.predict(array_double)
    except:
        print(filename)
    hist=np.histogram(a,bins=[0,1,2,3])
    #append the dataframe into the array 
    data.append(hist[0])
    k=k+1
    
#convert Array to Dataframe and append to the list
Output = pd.DataFrame(data)
#add row class 
Output["Class"] = i 
csv_data=Output.to_csv('finalFolder2.csv', mode='a',header=False,index=False)




#--------Displaying the features--------------------- 
#---------Displaying features for  folder1-------------
print("Fall")
dat1= pd.read_csv('finalFolder1.csv',header=None)
print(dat1)

#---------Displaying features for  folder1-------------
print("No fall")
dat2= pd.read_csv('finalFolder2.csv',header=None)
print(dat2)

#--------Merging Fall and No Fall data 
A = dat1.append(dat2)
A

#-----------save the predicted data into csv file---------
csv_data=A.to_csv('FinalF.csv', mode='a',header=False,index=False)