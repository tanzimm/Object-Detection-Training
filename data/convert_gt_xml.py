import sys
import os
import glob
import xml.etree.ElementTree as ET

x_factor = 378.0/416.0
y_factor = 504.0/416.0

def get_xml_gt(FILE):

  # make sure that the cwd() in the beginning is the location of the python script (so that every path makes sense)
  os.chdir(os.path.dirname(os.path.abspath(__file__)))

  # change directory to the one with the files to be changed
  parent_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
  parent_path = os.path.abspath(os.path.join(parent_path, os.pardir))
  GT_PATH = os.path.join(parent_path,'object detection','data',FILE)
  #print(GT_PATH)
  os.chdir(GT_PATH)

  # old files (xml format) will be moved to a "backup" folder
  ## create the backup dir if it doesn't exist already
  if not os.path.exists("backup"):
    os.makedirs("backup")

  # create VOC format files
  xml_list = glob.glob('*.xml')
  if len(xml_list) == 0:
    print("Error: no .xml files found in ground-truth")
    sys.exit()
  for tmp_file in xml_list:
    #print(tmp_file)
    # 1. create new file (VOC format)
    with open(tmp_file.replace(".xml", ".txt"), "a") as new_f:
      root = ET.parse(tmp_file).getroot()
      for obj in root.findall('object'):
        obj_name = obj.find('name').text
        bndbox = obj.find('bndbox')
        left = bndbox.find('xmin').text
        top = bndbox.find('ymin').text
        right = bndbox.find('xmax').text
        bottom = bndbox.find('ymax').text

        #needed for YOLO
        # left = int(bndbox.find('xmin').text)
        # top = int(bndbox.find('ymin').text)
        # right = int(bndbox.find('xmax').text)
        # bottom = int(bndbox.find('ymax').text)

        # left = str(round(left/x_factor))
        # right = str(round(right/x_factor))
        # top = str(round(top/y_factor))
        # bottom = str(round(bottom/y_factor))



        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
    # 2. move old file (xml format) to backup
    os.rename(tmp_file, os.path.join("backup", tmp_file))
  print("Conversion completed!")


if __name__ == '__main__':

  get_xml_gt("") # KEEP THIS UNCOMMENTED TS BEEN CALLED SOMEWHERE