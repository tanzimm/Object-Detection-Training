import os
import sys
import numpy as np
import shutil
from sys import exit

SAVE_CHECKPOINTS = 1000
CHECKPOINT_DIR = "./checkpoints"


file = open("map4.txt","r")
#all_mAP = open("map4.txt","w+")
lines = file.readlines()

for i in range(len(lines)):
    lines[i] = lines[i].rstrip()
    lines[i] = round(float(lines[i]),5)

best_map = max(lines)
best_cp= str(lines.index(best_map)*SAVE_CHECKPOINTS)


for filename in os.listdir(CHECKPOINT_DIR):

    name,ext = os.path.splitext(filename)

    if name[:11+len(best_cp)] == "model.ckpt-"+best_cp:

        name = filename[:10]
        ext = filename[11+len(best_cp):] 
  
        shutil.copy(CHECKPOINT_DIR  + "/" + filename, "./init/pre_trained_model/" + name+ext)


# for i in range(0,50,5):

#     all_mAP.write(lines[i])

#all_mAP.close()