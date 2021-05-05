import os, random
import shutil
from data.convert_gt_xml import get_xml_gt




def clear_dir(path):
  for filename in os.listdir(path):
    #print("removing: " + filename)
    
    if os.path.isdir(path+"/"+filename): 
      shutil.rmtree(path+"/"+filename)
    
    else: 
      os.remove(path + "/" + filename)



def data_split(train,test,GT,objs,train_split):
    
    clear_dir("./data/" + train)
    clear_dir("./data/" + test)
    clear_dir("./data/" + GT)

    for obj in objs:

        file_list = os.listdir("./data/" + obj)
        sample = round(len(file_list)*(train_split))

        for i in range(0,sample,1):
            move = random.choice(file_list)
            file_list.remove(move)
            print("train: " + move)
            shutil.copy("./data/"+ obj + "/" + move, "./data/" + train + "/" + obj + "/" + move)


        for file in file_list:
            print("test: " + file)
            shutil.copy("./data/"+ obj + "/" + file, "./data/" + test + "/" + file)



    get_xml_gt(test)
    os.chdir('..')
    print(os.getcwd())

    for filename in os.listdir("./" + test + "/"):
        if filename.endswith(".txt") or filename.endswith(".TXT"):
            shutil.move("./"+ test + "/" + filename, "./" + GT + "/" + filename)

    for filename in os.listdir("./" + test + "/backup"):
        
        shutil.move("./"+ test + "/backup/" + filename, "./" + test + "/" + filename)  


    shutil.rmtree("./" + test + "/backup")    




