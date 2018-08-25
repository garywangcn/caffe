#!/usr/bin/python
import os
import sys
import numpy as np

caffe_root="/home/gary/work/caffe/"

sys.path.insert(0, caffe_root + "python")
import caffe

MODEL_FILE = '/home/gary/work/caffe/examples/mnist/lenet.prototxt'
PRETRAINED = '/home/gary/work/caffe/examples/mnist/lenet_iter_10000.caffemodel'
IMAGE_FILE = 'test4.bmp'

input_image = caffe.io.load_image(IMAGE_FILE, color=False)
print "read image done"
net = caffe.Classifier(MODEL_FILE, PRETRAINED) 
print "create net done"
prediction = net.predict([input_image], oversample = False)
print "predict done"
caffe.set_mode_cpu()
print 'predicted class:', prediction[0].argmax()
