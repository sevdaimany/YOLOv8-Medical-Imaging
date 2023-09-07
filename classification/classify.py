from ultralytics import YOLO
import os
import numpy as np
import cv2 as cv

def train():
    model = YOLO("yolov8n-cls.yaml")
    model.train(data="D:\\computer-vision\\projects\\streamlit-dashboard\\classification\\Covid19-dataset", epochs=100)
    
    

def predict(img, st):
    model_path= os.path.join('.', 'runs', 'classify', 'train', 'weights', 'best.pt')
    model = YOLO(model_path)
    
    results = model.predict(img)
    result = results[0]
    
    class_names = result.names
    probs = result.probs.data.tolist()
    class_name = class_names[np.argmax(probs)].upper()
    width = img.shape[0]
    cv.putText(img, class_name, (width - 80, 60),cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv.LINE_AA)
    
    
    st.subheader('Output Image')
    st.image(img, channels="BGR", use_column_width=True)

