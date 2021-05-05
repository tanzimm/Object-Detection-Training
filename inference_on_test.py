import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import json
import time
from time import sleep,time
from utils import label_map_util
from utils import visualization_utils as vis_util
import xml.etree.ElementTree as ET




lm =    "/labelmap.pbtxt"

NUM_CLASSES = 3


class_dict = {
    1: "fork",
    2: "spoon",
    3: "grab",

}
def tf_init(model,graph):

    # Number of classes the object detector can identify
    # Load the label map.
    # Label maps map indices to category names, so that when our convolution
    # network predicts `5`, we know that this corresponds to `king`.
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    
    MODEL = model


    label_map = label_map_util.load_labelmap(MODEL+lm)
    #label_map = label_map_util.load_labelmap('init_files/dish_inf/dish_lm.pbtxt')
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(MODEL+graph, 'rb') as fid:
        #with tf.gfile.GFile('init_files/dish_inf/dish_inf.pb', 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    
    sess = tf.Session(graph=detection_graph)
    # Define input and output tensors (i.e. data) for the object detection classifier

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    return sess,image_tensor,detection_boxes,detection_scores,detection_classes,num_detections,category_index

def inference(frame,tf_ic,display=False):
        
    sess = tf_ic[0]
    image_tensor = tf_ic[1]
    detection_boxes = tf_ic[2]
    detection_scores = tf_ic[3]
    detection_classes = tf_ic[4]
    num_detections = tf_ic[5]
    category_index = tf_ic[6]

    frame_expanded = np.expand_dims(frame, axis=0)

    boxes,scores,classes,num = sess.run(
    [detection_boxes, detection_scores, detection_classes,num_detections],
    feed_dict={image_tensor: frame_expanded})

    if display == True:
        vis_util.visualize_boxes_and_labels_on_image_array(frame,np.squeeze(boxes),np.squeeze(classes).astype(np.int32),np.squeeze(scores),category_index,use_normalized_coordinates=True,line_thickness=8,min_score_thresh=0.5) 
        cv_show(0,1,1,frame)

    scores = (score for score in scores[0] if score*100 >= 1)

    #print(scores)

    return boxes,scores,classes,num


def create_GT(MODEL,img_path,labels,graph,sample=-1):

    print(graph)
 
    
    GT =  "./" + MODEL + "/detection results/"
    images = "./data/" + img_path + "/"
    image_labels = "./data/" + labels +  "/"

    tf_ic = tf_init(MODEL,graph)
   
    counter = 0
    total_files = 3434
    total_time = 0

    for filename in os.listdir(image_labels):
        #if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".JPG") or filename.endswith(".JPEG") or filename.endswith(".PNG"):

        tree = ET.parse(image_labels+filename)
        root = tree.getroot()
        frame_name = root.findtext("filename")
        

        name, _ = os.path.splitext(filename)

        name += ".txt"
        print(name)
        # if name[0:6] == "cs_img" or name[0:5] == "image":
        #     frame = cv2.imread("./data/additional images/"+frame_name)
        
        # else: 
        frame = cv2.imread(images+frame_name)
    
        file = open(GT+name,"w+")
        counter += 1   


        if graph != "yolo": 
            h = frame.shape[0]
            w = frame.shape[1]
            count = 0

            start = time ()

            boxes,scores,classes,num= inference(frame,tf_ic)

            total_time += (time() - start)
            
            
            for i in scores:
                count += 1
                _,rect_points = find_keypoints(boxes,count,h,w)
                min_x =int(rect_points[0][0])
                max_x =int(rect_points[1][0])
                min_y =int(rect_points[0][1])
                max_y= int(rect_points[1][1])

                
                file.write(class_dict[int(classes[0][count-1])])
                file.write(" ")
                file.write(str(i))
                file.write(" ")
                file.write(str(min_x))
                file.write(" ")
                file.write(str(min_y))
                file.write(" ")
                file.write(str(max_x))
                file.write(" ")
                file.write(str(max_y))
                file.write("\n")

    
        file.close()
        #cv_show(0,1,1,frame)
        #print(counter)

        if counter == sample:
            break

    print(total_time)
    print(total_time/total_files)




def cv_show(cvt, wait, dst,frame):

    if cvt == True:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR )
           
 
    cv2.imshow("Image", frame)
    if wait == True:
        cv2.waitKey(0)

    if dst == True:
        cv2.destroyAllWindows()

def cv_circle (points,frame,display=False):


    if len(points) == 1:

        x = int(points[0][0])
        y = int(points[0][1])

        cv2.circle(frame,(x,y),3,(0,0,255), -1)
     

    elif len(points) > 1:   
        for i in range(len(points)):

            x = int(points[i][0])
            y = int(points[i][1])

            cv2.circle(frame,(x,y),3,(0,0,255), -1)

    if display == True:
        cv_show(0,1,1,frame)

    return frame

def cv_gray(frame,display=False):


    gray = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2GRAY)

    if display == True:
        cv_show(0,1,1,gray)

    return gray


def cv_rectangle (bbox,frame,display=False,color=(0,255,0)):

    x,y,w,h = bbox

    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)

    cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)

    if display == True:
        cv_show(0,1,1,frame)

    return frame

def find_keypoints(boxes,count,h,w):

    x1 = boxes[0][count-1][1]*w
    y1 = boxes[0][count-1][0]*h
    x2 = boxes[0][count-1][3]*w
    y2 = boxes[0][count-1][2]*h

    xCenter = int((x1+x2)/2) #xmin + xmax
    yCenter = int((y1+y2)/2) #ymin + ymax

    dx = abs(x2-x1)
    dy = abs(y2-y1)

    p1 = [x1,y1]
    p4 = [x2,y2]


    return np.asarray([xCenter,yCenter]),[p1,p4,dx,dy]


# tf_ic = tf_init("new faster rcnn og")
# image = cv2.imread("t6_Color.png")
# frame = cv2.resize(image,(640, 480), interpolation=cv2.INTER_AREA)
# inference(frame,tf_ic,True)