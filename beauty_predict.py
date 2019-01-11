# coding=utf-8

# 预测人脸颜值

from keras.layers import Conv2D, Input, MaxPool2D,Flatten, Dense, Permute, GlobalAveragePooling2D,Dropout
from keras.applications.resnet50 import ResNet50,Dense
from keras.models import Model,Sequential
from keras.optimizers import adam
import numpy as np
import pickle
import keras
import cv2
import sys
import dlib
import os
import tensorflow as tf 

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(APP_ROOT, "models/mmod_human_face_detector.dat")
cnn_face_detector = dlib.cnn_face_detection_model_v1(model_path)



class Beauty_Predict():

    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        resnet = ResNet50(include_top=False, pooling='avg')
        self._model = Sequential()
        self._model.add(resnet)
        self._model.add(Dense(5, activation='softmax'))
        self._model.layers[0].trainable = False

        self._model.load_weights(os.path.join(APP_ROOT, 'models/model-ldl-resnet.h5'))


    def pred_beauty(self, face_img):

        resized_im = cv2.resize(face_img, (224, 224))
        normed_im = np.array([(resized_im - 127.5) / 127.5])

        pred = self._model.predict(normed_im)
        ldList = pred[0]
        out = 1 * ldList[0] + 2 * ldList[1] + 3 * ldList[2] + 4 * ldList[3] + 5 * ldList[4]

        out = self.score_mapping(out)

        return out


    # 1-5  to 40-100
    def score_mapping(self, modelScore):

        if modelScore <= 2:
            mapping_score = (modelScore - 1.0) / (2.0 - 1.0) * (60 - 40) + 40

        elif modelScore <= 3:
            mapping_score = (modelScore - 2.0) / (3.0 - 2.0) * (85 - 70) + 70

        elif modelScore <= 4:
            mapping_score = (modelScore - 3.0) / (4.0 - 3.0) * (90 - 80) + 80

        elif modelScore <= 5:
            mapping_score = (modelScore - 4.0) / (5.0 - 4.0) * (100 - 90) + 90

        return mapping_score



if __name__ == "__main__":

    c_beauty_predict = Beauty_Predict()

    img_path = os.path.join(APP_ROOT, "images") + "/fbb.jpg"
    print (img_path)
    im0 = cv2.imread(img_path)

    if im0.shape[0] > 1280:
        new_shape = (1280, im0.shape[1] * 1280 / im0.shape[0])
    elif im0.shape[1] > 1280:
        new_shape = (im0.shape[0] * 1280 / im0.shape[1], 1280)
    elif im0.shape[0] < 640 or im0.shape[1] < 640:
        new_shape = (im0.shape[0] * 2, im0.shape[1] * 2)
    else:
        new_shape = im0.shape[0:2]

    im = cv2.resize(im0, (int(new_shape[1]), int(new_shape[0])))
    dets = cnn_face_detector(im, 0)

    for i, d in enumerate(dets):

        face = [d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()]
        croped_im = im[face[1]:face[3], face[0]:face[2], :]

        score = c_beauty_predict.pred_beauty(croped_im)
        print (score)


