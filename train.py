# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Binary to run train and evaluation on object detection model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from time import time, sleep
import tensorflow as tf
from inference_on_test import create_GT
from map import find_map
import model_hparams
import model_lib
import os
from create_tfrecord import create_tf_records_from_csv
from split_data import data_split

from google.protobuf import text_format
from object_detection import exporter
from object_detection.protos import pipeline_pb2
import shutil
from sys import exit


flags.DEFINE_string(
    'model_dir', None, 'Path to output model directory '
    'where event and checkpoint files will be written.')
flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config '
                    'file.')
flags.DEFINE_integer('num_train_steps', None, 'Number of train steps.')
flags.DEFINE_boolean('eval_training_data', False,
                     'If training data should be evaluated for this job. Note '
                     'that one call only use this in eval-only mode, and '
                     '`checkpoint_dir` must be supplied.')
flags.DEFINE_integer('sample_1_of_n_eval_examples', 1, 'Will sample one of '
                     'every n eval input examples, where n is provided.')
flags.DEFINE_integer('sample_1_of_n_eval_on_train_examples', 5, 'Will sample '
                     'one of every n train input examples for evaluation, '
                     'where n is provided. This is only used if '
                     '`eval_training_data` is True.')
flags.DEFINE_string(
    'hparams_overrides', None, 'Hyperparameter overrides, '
    'represented as a string containing comma-separated '
    'hparam_name=value pairs.')
flags.DEFINE_string(
    'checkpoint_dir', None, 'Path to directory holding a checkpoint.  If '
    '`checkpoint_dir` is provided, this binary operates in eval-only mode, '
    'writing resulting metrics to `model_dir`.')
flags.DEFINE_boolean(
    'run_once', False, 'If running in eval-only mode, whether to run just '
    'one round of eval vs running continuously (default).'
)

FLAGS = flags.FLAGS


def calculate_map(graph, single=False, delete_inf=True, cp=-1, iou=0.5):

    max_checkpoints = -10
    i = 0
    file_list = os.listdir("./" + MODEL + CHECKPOINT_DIR)

    for filename in file_list:
        name, ext = os.path.splitext(filename)

        if name[:10] == "model.ckpt":
            if int(name[11:]) > max_checkpoints:
                max_checkpoints = int(name[11:])

    if single == False:

        i = round(max_checkpoints/((len(file_list)/3)-1))
        print("Calculating mAP for all checkpoints")
        print(max_checkpoints, i)

        all_mAP = open(MODEL + TXT, "w+")

        for checkpoint_num in range(0, max_checkpoints, i):

            export_inference_graph(checkpoint_num, INF_OUTPUT)
            clear_dir(path="./" + MODEL + "/detection results")
            create_GT(MODEL, IMG_PATH, TEST, graph)
            mAP = find_map(MODEL, checkpoint_num, "output", GT, iou)
            all_mAP.write(str(mAP))
            all_mAP.write("\n")

            path = os.path.join(os.getcwd(), MODEL, "inference_graph")
            shutil.rmtree(path)

        all_mAP.close()

    else:
        print("Calculating mAP for max checkpoint only")
        if cp != -1:
            max_checkpoints = cp

        print(max_checkpoints)


        export_inference_graph(max_checkpoints, INF_OUTPUT)
        clear_dir(path="./" + MODEL + "/detection results")
        create_GT(MODEL, IMG_PATH, TEST, graph)
        mAP = find_map(MODEL, max_checkpoints, "/none", GT, iou)
        print(mAP)

        if delete_inf == True:
            path = os.path.join(os.getcwd(), MODEL, "inference_graph")
            shutil.rmtree(path)

        return mAP



def clear_dir(path, keep_dir=False):

    if os.path.isdir(path):

        for filename in os.listdir(path):
            #print("removing: " + filename)

            if os.path.isdir(path+"/"+filename):
                if keep_dir == False:
                    shutil.rmtree(path+"/"+filename)

            else:
                os.remove(path + "/" + filename)


def export_inference_graph(checkpoint_num, output, move=False):

    tf.reset_default_graph()

    if checkpoint_num == -2:
        trained_checkpoint_prefix = MODEL + "/inference_best/model.ckpt"

    elif checkpoint_num == -3:
        trained_checkpoint_prefix = MODEL + "/init/pre_trained_model/model.ckpt"
    else:
        trained_checkpoint_prefix = MODEL + CHECKPOINT_DIR + \
            "/model.ckpt-" + str(checkpoint_num)

    if os.path.isdir("./" + MODEL + output):
        clear_dir(path="./" + MODEL + output)

    print(trained_checkpoint_prefix)
    #_ = input("waiting")

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile(MODEL+CONFIG, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)
    text_format.Merge('', pipeline_config)
    if None:
        input_shape = [
            int(dim) if dim != '-1' else None
            for dim in None.split(',')
        ]
    else:
        input_shape = None
    exporter.export_inference_graph(
        'image_tensor', pipeline_config, trained_checkpoint_prefix,
        MODEL+output, input_shape=input_shape,
        write_inference_graph=False)

    print("inference graph  created")
    print(trained_checkpoint_prefix)

    if move == True:
        best_cp = str(checkpoint_num)
        for filename in os.listdir("./" + MODEL + output):

            name, ext = os.path.splitext(filename)

            if name[:10] == "model.ckpt":
                os.remove("./" + MODEL + output + "/" + filename)

        for filename in os.listdir("./" + MODEL+CHECKPOINT_DIR):

            name, ext = os.path.splitext(filename)

            if name[:11+len(best_cp)] == "model.ckpt-"+best_cp:

                name = filename[:10]
                ext = filename[11+len(best_cp):]

                shutil.copy("./" + MODEL + CHECKPOINT_DIR + "/" +
                            filename, "./" + MODEL + output + "/" + name+ext)


def move_checkpoints(temp_dir):

    path_dir = temp_dir

    for filename in os.listdir(path_dir):
        name, ext = os.path.splitext(filename)

        if name[:10] == "model.ckpt":

            shutil.move(path_dir + "/" + filename, MODEL +
                        CHECKPOINT_DIR + "/" + filename)

    for filename in os.listdir(path_dir):

        name, ext = os.path.splitext(filename)

        if name == "checkpoint":
            os.remove(path_dir + "/"+filename)
        elif name[:6] == "events":
            os.remove(path_dir + "/"+filename)
        elif name[:5] == "graph":
            os.remove(path_dir + "/"+filename)


def find_best_map():

    file = open("./" + MODEL + TXT, "r")

    lines = file.readlines()

    for i in range(len(lines)):
        lines[i] = lines[i].rstrip()
        lines[i] = round(float(lines[i]), 5)

    best_map = max(lines)
    best_cp = str(lines.index(best_map)*SAVE_CHECKPOINTS)

    if os.path.isdir("./" + MODEL + INF_OUTPUT):
        clear_dir(path="./" + MODEL + INF_OUTPUT)

    export_inference_graph(best_cp, INF_OUTPUT, move=True)

    return best_map*100, best_cp


def get_coco_map(cp):

    file = open(MODEL + "/coco_map.txt", "w+")

    map_50_95 = 0
    map_50 = 0
    map_75 = 0
    mAP = 0
    
    graph = INF_OUTPUT +"/frozen_inference_graph.pb"

    clear_dir(path="./" + MODEL + "/detection results")
    create_GT(MODEL, IMG_PATH, TEST, graph)

    print("starting")

    for i in range(50, 100, 5):

        print(i)
        iou = (i/100)
        mAP = find_map(MODEL, 200, "/none", GT, iou)
        map_50_95 += mAP

        if i == 50:
            map_50 = mAP
            file.write(str(map_50))
            file.write("\n")

        elif i == 75:
            map_75 = mAP
            file.write(str(map_75))
            file.write("\n")

    map_50_95 /= 10
    file.write(str(map_50_95))
    file.write("\n")

    print("Final Map Values: ")
    print(map_50*100)
    print(map_75*100)
    print(map_50_95*100)
    file.close()


def main(unused_argv):

    #while saving checkpoints tensorflow has a weird error on windows
    #so temporarily save checkpoints somewhere on desktop, this program will later move the checkpoints onto the proper directory
    TEMP_DIR = ""

    clear_dir(TEMP_DIR, True)
    config = tf.estimator.RunConfig(model_dir=TEMP_DIR,
                                    save_checkpoints_steps=SAVE_CHECKPOINTS, keep_checkpoint_max=None)

    train_and_eval_dict = model_lib.create_estimator_and_inputs(
        run_config=config,
        hparams=model_hparams.create_hparams(FLAGS.hparams_overrides),
        pipeline_config_path=MODEL+CONFIG)
   

    estimator = train_and_eval_dict['estimator']
    train_input_fn = train_and_eval_dict['train_input_fn']
    eval_input_fns = train_and_eval_dict['eval_input_fns']
    eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
    predict_input_fn = train_and_eval_dict['predict_input_fn']
    train_steps = train_and_eval_dict['train_steps']

    if FLAGS.checkpoint_dir:
        if FLAGS.eval_training_data:
            name = 'training_data'
            input_fn = eval_on_train_input_fn
        else:
            name = 'validation_data'
            # The first eval input will be evaluated.
            input_fn = eval_input_fns[0]
        if FLAGS.run_once:
            estimator.evaluate(input_fn,
                               steps=None,
                               checkpoint_path=tf.train.latest_checkpoint(
                                   FLAGS.checkpoint_dir))
        else:
            model_lib.continuous_eval(estimator, FLAGS.checkpoint_dir, input_fn,
                                      train_steps, name)
    else:
        train_spec, eval_specs = model_lib.create_train_and_eval_specs(
            train_input_fn,
            eval_input_fns,
            eval_on_train_input_fn,
            predict_input_fn,
            TRAIN_STEPS,
            eval_on_train_data=False)

        # Currently only a single Eval Spec is allowed.

        tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])

        #moving checkpoints to the proper directory, it will delete from the temp_dir directory
        move_checkpoints(TEMP_DIR)  

        #calculating map for all checkpoints and output map value on text file located inside the model directory
        calculate_map(GRAPH)

        #it will go through the text file to determine which checkpoint had the best map value
        best_map, best_cp = find_best_map()

        #it will calcualte the coco map values for the best checkpoint and output that into a text file inside model directory 
        get_coco_map(best_cp)

        print("\n")
        print("Best map: " + str(best_map))
        print("Checkpoint : " + best_cp)




if __name__ == '__main__':

    #will create a inference graph directory inside model directory 
    INF_OUTPUT = "/inference_graph"

    #text file containing the map value of all checkpoints
    TXT = "/map_complete.txt"

    #put all your image files in the data/images folder 
    IMG_PATH = "images"

    #put all your respective annotations in these folders 
    GT = "test GT" 
    TEST = "test"
    TRAIN = "train"
    VAL = "val"

    #line 106: put path of fine tuned checkpoint
    #line 130: put path of train TF record 
    #line 132 and 146: put path of label map
    #line 142: put path of val TF record
    #download pre trained models from:
    #https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md
    CONFIG = "/init/faster_rcnn_inception_v2_petso.config"

    OBJS = ["images_seperated/class1","images_seperated/class2","images_seperated/class3"] #in the data folder seperate your images data into its different classes and place them in different folders
    TRAIN_SPLIT = 0.7 #70 train 30 test 

    TRAIN_STEPS = 40000 #total steps
    SAVE_CHECKPOINTS = 1000 #save every N checkpoints
    CHECKPOINT_DIR = "checkpoint"

    GRAPH = INF_OUTPUT + "/frozen_inference_graph.pb"

    MODEL = "Faster RCNN" #can change to a different model, just find the proper config and pre trained model from official TensorFlow Obect Detection API 

    #will automatically split your data and generate the ground truth test file to calculate mAP after training 
    data_split(TRAIN, TEST, GT, OBJS, TRAIN_SPLIT)
    #creates the TF records required for training 
    create_tf_records_from_csv(TRAIN, VAL, MODEL, IMG_PATH)
    #trains the models and calculats MAP for all checkpoints
    #it will find the checkpoint with the highest mAP and then calculate the coco map values for it 
    tf.app.run() 
