from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf
import glob
import pandas as pd
import xml.etree.ElementTree as ET
from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict
from distribution_weight import distribution_weight
import csv
from tempfile import NamedTemporaryFile
import shutil

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def class_text_to_int(row_label):
    if row_label == 'fork':
        return 1
    elif row_label == 'spoon':
        return 2
    elif row_label == 'grab':
        return 3
    else:
        return 0


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def create_tf_example_weights(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    weights = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        weights.append(row['distribution'])
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/weight': dataset_util.float_list_feature(weights),
    }))
    return tf_example



def create_tf_records_from_csv(train,val,model,images):


    path = "C:/Users/tanzimmashrur/OneDrive - University of Guelph/Masters OneDrive/Object detection" #if object detection folder moves this needs to updated
    save = path + "/" + model + "/init/"


    count = 0 
    
    for folder in [train,val]:
        image_label = os.path.join(path,'data',folder)
        xml_df = xml_to_csv(image_label)


        if count == 0 :
            xml_df.to_csv((save + "train" + '_labels.csv'), index=None)
            print('SUCCESSFULLY converted xml to csv.')
            #distribution_weight(save + "train" + '_labels.csv',1,1000)
            #adjust_weight(save + "train" + '_labels.csv',image_label,1000)
            image_path = "./data/" + images
            writer = tf.python_io.TFRecordWriter(save + "train.record")


        elif count== 1:
            xml_df.to_csv((save + "val" + '_labels.csv'), index=None)
            print('SUCCESSFULLY converted xml to csv.')
            #distribution_weight(save + "val" + '_labels.csv',1,1000)
            #adjust_weight(save + "val" + '_labels.csv',image_label,1000)
            image_path = "./data/" + images
            writer = tf.python_io.TFRecordWriter(save + "val.record")
        

        if count == 0:
            f ="train"
        elif count == 1:
            f ="val"
    
        examples = pd.read_csv(save + f + '_labels.csv')
        grouped = split(examples, 'filename')
        for group in grouped:
            tf_example = create_tf_example(group, image_path)
            writer.write(tf_example.SerializeToString())

        writer.close()
        print('Successfully created the TFRecords')
        count += 1


def adjust_weight(file,src,weight):
    
    
    for filename in os.listdir(src):

        name, _ = os.path.splitext(filename)
      
        if name[:2] == "d2":
            tree = ET.parse(src+"/"+filename)
            root = tree.getroot()
            frame_name = root.findtext("filename")
            update_csv(frame_name,file,weight)

        

def update_csv(frame_name,file,weight):

    fields = ["filename",  "width",   "height",  "class",   "xmin",    "ymin",  "xmax",  "ymax", "distribution"]

    tempfile = NamedTemporaryFile(mode='w', delete=False, newline='')

    with open(file, 'r') as csvfile, tempfile:

        reader = csv.DictReader(csvfile, fieldnames=fields)
        writer = csv.DictWriter(tempfile, fieldnames=fields)
                
        for row in reader:
    
            if row['filename'] == frame_name:
                print(frame_name)
                row['distribution'] = weight
                row = {'filename': row['filename'], 'width': row['width'], 'height': row['height'], 'class': row['class'], 'xmin': row['xmin'], 'ymin': row['ymin'], 'xmax': row['xmax'],'ymax': row['ymax'], 'distribution': row['distribution']}
                writer.writerow(row)
            else:
                writer.writerow(row)

    shutil.move(tempfile.name, file)




# file = "./PT2/init/train_labels.csv"
# weight = 1000
# src = "./data/daily train/train/"


# adjust_weight(file,src,weight)

            # i = 0
            # with open(file) as fp:
            #     lines = fp.readlines()

            # with open(file, 'w') as f:
               
            #     for line in lines:
            #         if (i > 0):
            #             line = line.rstrip()

            #             if (line.startswith(frame_name)):
            #                 weight = weight2

            #                 # separate the string by commas and update the last element
            #                 wline = line.split(",")
            #                 wline[len(wline) - 1] = str(weight)

            #                 # join the string back together with the updated value 
            #                 line = ','.join(wline)
            #                 line += "\n"

            #                 f.write(line)
            #                 break
            #         i += 1



# model = "PT2"
# path = "C:/Users/tanzimmashrur/OneDrive - University of Guelph/Masters OneDrive/Object detection" #if object detection folder moves this needs to updated
# save = path + "/" + model + "/init/"
# distribution_weight(save + "train" + '_labels.csv',1,1000)

# src = "./data/daily train/temp/"

# counter = 2

# for filename in os.listdir(src):
#     if filename.endswith(".xml") or filename.endswith(".XML"):

        # os.rename(src+filename,src+"d2-"+str(counter)+".xml")
        # counter += 1
        # name, _ = os.path.splitext(filename)

        # if name[:2] == "d1":
        #     print(name)



# image_label = os.path.join(path,'data',"daily train","temp")
# xml_df = xml_to_csv(image_label)

# xml_df.to_csv((save + "train" + '_labels.csv'), index=None)
# print('SUCCESSFULLY converted xml to csv.')
# distribution_weight(save + "train" + '_labels.csv',1,1000)





