import streamlit as st
import pickle
from PIL import Image
import os
import cv2
import numpy as np
import pandas as pd

with open("./css/app.css", "r") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

model_dir = r"./trained models/pedestrian/"

models = {
    "KNN": pickle.load(open(os.path.join(model_dir, "model_knn.sav"), "rb")),
    "lr_model": pickle.load(open(os.path.join(model_dir, "Model_LOG_REG.sav"), "rb")),
    "svm_model": pickle.load(open(os.path.join(model_dir, "PCA_3_Model.sav"), "rb")),
    "rf_model": pickle.load(open(os.path.join(model_dir, "model_rf.sav"), "rb")),
    "svc_model": pickle.load(open(os.path.join(model_dir, "SVC_RBF_PCA_3.sav"), "rb")),
    "dt_model": pickle.load(open(os.path.join(model_dir, "model1.sav"), "rb")),
    "kmeans_model": pickle.load(open(os.path.join(model_dir, "Kmeans_CL_5_Model.sav"), "rb"))
}

# models_options = ["KNN", "lr_model", "svm_model", "rf_model", "svc_model", "dt_model", "kmeans_model"]
# models_display = ["KNN", "Logistic Regression", "SVM", "Random Forest", "SVC", "Decision Tree", "K-Means"]
models_options = ["KNN", "kmeans_model"]
models_display = ["KNN", "K-Means"]

# def workKNN()


def setModel(model):
    return models[model]

def processImg(model, img_path):
    st.write(model)
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
            a = setModel(model).predict(array_double)
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

def main(model, img_path):
    img = processImg(model, img_path).tolist()
    # print(img)
    return pd.DataFrame([
        ["Pedestrian", img.count(1)/(img.count(0)+img.count(1))*100],
        ["Other", img.count(0)/(img.count(0)+img.count(1))*100]
    ], columns=["Sign", "Probablity (%)"])

if __name__ == '__main__':
    st.title("Traffic Sign Detection and Recognition")

    model = st.selectbox(
        "Select Model", 
        models_options,
        format_func=lambda x: models_display[models_options.index(x)]
    )

    file = st.file_uploader("Upload the image", type=["jpg", "png"], label_visibility="collapsed")

    if st.button("Submit"):
        if file is None:
            st.warning("Please upload an image")

        if file is not None:
            try:
                data = file.getvalue()

                with open(f"./uploaded/{file.name}", "wb") as f:
                    f.write(data)
                
                result = main(model, f"./uploaded/{file.name}")
                st.image(Image.open(f"./uploaded/{file.name}"), width=400)

                st.table(result)

            except Exception as e:
                st.write("Something went wrong..! Please try again..!")
                st.write("Error: ", e)
