#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pickle
from PIL import Image
import os
import cv2
import numpy as np
import pandas as pd


# In[4]:


model_dir = r"C:\Users\SAI\Desktop\Traffic Sign Detectiona and Recognition\trained models\pedestrian"

knn_model = pickle.load(open(os.path.join(model_dir, "model_knn.sav"), "rb"))


# In[5]:


# @app.route("/")
def welcome():
    return "Welcome to Traffic Sign Detection and Recognition"


# In[6]:


def processImg(img_path):
    print(img_path)
    i = cv2.imread(img_path)
    cv2.imshow("Image", i)
    cv2.waitKey(0) 
  
    #closing all open windows 
    cv2.destroyAllWindows() 
    try:
        path = os.path.join(img_path)
        a = cv2.imread(path)
        resize = (280,430)
        img = cv2.resize(a, resize)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        out = pd.DataFrame(descriptors)
        
        try:
            array_double = np.array(out, dtype=np.double)
            a = knn_model.predict(array_double)
            hist=np.histogram(a,bins=[0,1,2,3,4,5])
            print("HISTOGRAM",hist)
            return a
        except ValueError as v:
            print("Something went wrong..!")
            print("Error: ", v)
            return "Error"
        
    except Exception as e:
        print(str(e))
        return "Error"


# In[7]:


def main(img_path):
    img = processImg(img_path).tolist()
    if (img.count(0) > img.count(1)):
        st.write("No Pedestrian")
    else:
        st.write("Pedestrian")
    return


# In[8]:


main(r'C:/Users/SAI/Desktop/Traffic Sign Detectiona and Recognition/dataset/Pedestrian_2/pedestrian2.jpg')

