from inference_on_test import create_GT
from map import find_map
from train import export_inference_graph,clear_dir
import os
import shutil
import tensorflow as tf



MODEL = "new faster rcnn og"
#MODEL = "new yolo"

GT = "complete test GT"
#GT = "temp_test"
IMG_PATH = "images"
TEST = "complete test"



def get_coco_map(cp,file):

    file = open(file,"w+")

    map_50_95= 0
    map_50 = 0
    map_75 = 0
    mAP = 0

    # # if MODEL != "yolo_v3" :
    output = "/old_dis_inf (BEST)"
    graph =  output + "/frozen_inference_graph.pb"
    #export_inference_graph(cp,output)
    # else:
    #     graph = "yolo"

    sample = -1
    clear_dir(path ="./" + MODEL + "/detection results")
    create_GT(MODEL,IMG_PATH,TEST,graph,sample)


    # for filename in os.listdir("./" + MODEL + "/detection results"):
    #     #name, _ = os.path.splitext(filename)
    #     print("moving " + filename)
    #     shutil.copy("./data/test GT/" + filename,"./data/temp_test/" +filename)


    print("starting")

    for i in range(50,100,5):

        print(i)
        iou= (i/100)
        mAP = find_map(MODEL,200,"/none",GT,iou)
        map_50_95 += mAP

        if i == 50: 
          map_50 = mAP
          file.write(str(map_50*100))
          file.write("\n")

        elif i == 75:
          map_75 = mAP
          file.write(str(map_75*100))
          file.write("\n")


    map_50_95 /= 10
    file.write(str(map_50_95*100))
    file.write("\n")

    print("Final Map Values: ")
    print(map_50*100)
    print(map_75*100)
    print(map_50_95*100)
    file.close()


print("MODEL: " + MODEL)
get_coco_map(20000, MODEL  + "/best_coco_map_w_new_distribution.txt")



