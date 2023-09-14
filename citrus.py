import OS
import cv2
import numpy as np
from picamera2 import Picamera2
import tensorflow as tf
import argparse
import sys
import datatime
from time import sleep

Good_Citrus=0
Bad_Citrus=0

IM_WIDTH = 680
IM_ HEIGHT = 620

camera_type =  'picamera'
parser = argparse.ArgumentParser()
args = paerser.parse_args()

sys.path.append('..')

from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME= 'ssdlite_mobilenet_v2_coco_2018_05_09'

CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
ATH_TO_LABELS =  os.path.join(CWD_PATH,'data','citrus.pbtxt')

NUM_CLASEES = 90

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label)_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()

with detection_graph.as_default():
  od_graph_def = tfGraphDef()
  with tf.gfileGFile(PATH_TO_CKPT, 'rb' as fid:
    serialized_graph  = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

  sess= tf.Session(graph=detection_graph)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

num_detections = detection_graph.get_tensor_by_name('num_detections:0')

frame_rate_calc=1
freq = cv2.getTickFrequency ()
font = cv2.FONT.HERSHEY.SIMPLEX

if camera_type == 'picamera'
    camera = PiCamera()
    camera.resolution = (IM_WIDTH, IM_HEIGHT)
    camera.framerate = 10
    rawCapture = PiRGBArray (camera, size=(IM_WIDTH,IM_HEIGHT)
    rawCapture = truncate (0)

    for frame1 in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True)
        t1 = cv2.getTickCount()

        frame = np.copy(frame1.array)
        frame.setflags(write=1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_expanded = np.expand_dims(frame_rgb, axis=0)

        (boxes, scores, classes, num) =sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: frame_expanded})

  vis_util.visualize_boxes_and_labels_on_image_array(
      frame,
      np.squeeze(boxes),
      np.squeeze(classes).astype(np.int32),
      np.squeeze(score),
      category_index,
      use_norma;ized_coordinates=True,
      line_thickness=8,
      min_score_thresh=0.40)

  cv2.putText(frame,"FPS: {0:.2f}.format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)
  name= str([category_index.get(value) for index,value in enumerate(classes[0]) if score[0,index] > 0.5])
  print (name)
  substring1 = (" 'GOOD CITRUS' ')
  substring2 = (" 'BAD CITRUS' ")
  count1 = name.count(substring1)
  count2 = name.count(substring2)
  print (count1)
  print (count2)
  print ("GOOD CITRUS=", end=' ')
  print (Good_Citrus, end=' ')
  print ("BAD CITRUS==", end= ' ')
  print (Bad_Citrus)
  if (count1 == 1):
      Good_Citrus = Good_Citrus+1
      print(Good_Citrus)
      if (count1 > 1):
          Good_Citrus = Good_Citrus+count1
  if (count2 == 1):
      Bad_Citrus = Bad_Citrus+1
      print (Bad_Citrus)
  if (count2 > 1):
        Bad_Citrus = Bad_Citrus+count2
else:
  print("NOT CITRUS")

cv2.imshow('Object Detector', frame)

t2 = cv2.getTickCount()
time1 = (t2-tl)/freq
frame_rate_calc = 1/timel

if cv2.waitkey(1) == ord ('q'):
    break

  rawCapture.truncate(0)
camera.close()
cv2.destroyAllWindows()
