import os
import shutil
import cv2
import splitfolders



def split_train_test_val():
    input_dir = 'segmentation/data_'
    output_dir = "segmentation/data/"
    try:
        shutil.rmtree(output_dir)
    except:
        print('Folder not deleted')
    
    splitfolders.ratio(input_dir, output=output_dir, seed=1337, ratio=(.8, 0.1,0.1)) 
    try:
        shutil.rmtree(input_dir)
        print('Folder and its content removed') # Folder and its content removed
    except:
        print('Folder not deleted')
    
    

def masks_to_polygons():
    input_dir = 'segmentation/Dataset_BUSI_with_GT'
    output_dir_images = 'segmentation/data_/images'
    output_dir_masks = 'segmentation/data_/labels'
    
    try:
        shutil.rmtree(output_dir_images)
        shutil.rmtree(output_dir_masks)
        print('Folder and its content removed') # Folder and its content removed
    except:
        print('Folder not deleted')
                    
    os.makedirs(output_dir_images)
    os.makedirs(output_dir_masks)
     
    
    for i in os.listdir(input_dir):
        class_id = 0
        if i == "benign":
            class_id = 0
        elif i == "malignant":
            class_id = 1
        elif i =="normal":
            class_id = 2
        else:
            continue
        for j in os.listdir(os.path.join(input_dir, i)):
            
            output_file = ""
            if j[-8:-4] == "mask" or  j[-10:-6] == "mask":
                
                if j[-10:-6] == "mask":
                    output_file = os.path.join(output_dir_masks, j)[:-11]
                else:
                    output_file = os.path.join(output_dir_masks, j)[:-9]
                
                
                image_path = os.path.join(os.path.join(input_dir, i), j)
                # load the binary mask and get its contours
                mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

                H, W = mask.shape
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # convert the contours to polygons
                polygons = []
                for cnt in contours:
                    if cv2.contourArea(cnt) > 200:
                        polygon = []
                        for point in cnt:
                            x, y = point[0]
                            polygon.append(x / W)
                            polygon.append(y / H)
                        polygons.append(polygon)

                # print the polygons
                with open('{}.txt'.format(output_file), 'a') as f:
                    for polygon in polygons:
                        for p_, p in enumerate(polygon):
                            if p_ == len(polygon) - 1:
                                f.write('{}\n'.format(p))
                            elif p_ == 0:
                                f.write('{} {} '.format(class_id, p))
                            else:
                                f.write('{} '.format(p))

                    f.close()

            else:
                image_path = os.path.join(os.path.join(input_dir, i), j)
                shutil.copy(image_path, output_dir_images)




                    
                

