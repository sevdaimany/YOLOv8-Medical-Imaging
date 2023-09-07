from ultralytics import YOLO
import os
import cv2 as cv
from PIL import Image

def train():
    model = YOLO('yolov8n.yaml')  # build a new model from scratch
    model.train(data="D:\\computer-vision\\projects\\streamlit-dashboard\\detection\\data\\data.yaml", epochs=100)  # train the model

    # or you can run following in command line:
    # yolo detect train data=data.yaml model="yolov8n.yaml" epochs=1
    

def predict(img, confidence, st):
    # detection model
    model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'best.pt')
    model = YOLO(model_path)
     
     # Predict
    results = model.predict(img, conf=confidence)
    result = results[0]
    
    print("\n[INFO] Numer of objects detected : ", len(result.boxes) )
    
    
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        # im.show()  # show image
        # im.save('results.jpg')  # save image
        
    
    # OR
        
    # for obj in result.boxes.data.tolist():
    #     x1, y1, x2, y2, score, class_id = obj
        
    #     cv .rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 4)
    #     cv.putText(img, result.names[int(class_id)].upper(),  (int(x1), int(y1 - 10)),
    #                 cv.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv.LINE_AA)
        
            
    st.subheader('Output Image')
    st.image(im, channels="BGR", use_column_width=True)

        
    