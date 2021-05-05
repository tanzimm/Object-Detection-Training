from inference_on_test import  create_GT
from train import export_inference_graph
from map import find_map
import os
import sys
import numpy
import cv2
import shutil
from time import sleep,time


MODEL = "new faster rcnn og"
CHECKPOINT_DIR ="/checkpoints"
OUTPUT = "output2"
TXT = "/map4.txt"


images = "./data/images/"
#max_checkpoints = 50000
#i = 5000
#starting = 0


def calculate_map(max_only=False):


    max_checkpoints = -10
    i = 0
    file_list = os.listdir("./" + MODEL + CHECKPOINT_DIR)

    for filename in file_list:
        name,ext = os.path.splitext(filename)

        if name[:10] == "model.ckpt":
          if int(name[11:]) > max_checkpoints :
            max_checkpoints = int(name[11:])

   
    i = round(max_checkpoints/((len(file_list)/3)-1))    

    
    print(max_checkpoints,i)

    all_mAP = open(MODEL + TXT,"w+")
    start = time()

    for checkpoint_num in range (0,max_checkpoints,i):

        export_inference_graph(MODEL,CHECKPOINT_DIR,False,checkpoint_num)
        create_GT(MODEL)
        mAP = find_map(MODEL,checkpoint_num,OUTPUT)
        all_mAP.write(str(mAP))
        all_mAP.write("\n")

        path = os.path.join(os.getcwd(),MODEL,"inference_graph")
        shutil.rmtree(path)



    all_mAP.close()

    print(time()-start)

    
    



