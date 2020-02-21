#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Sample code to generate caption using beam search
'''
import sys
import json
import os
# comment out the below if you want to do type check. Remeber this have to be done BEFORE import chainer
#os.environ["CHAINER_TYPE_CHECK"] = "0"
import chainer 
import time

import argparse
import numpy as np
import math
from chainer import cuda
import chainer.functions as F
from chainer import cuda, Function, gradient_check, Variable, optimizers
from chainer import serializers

sys.path.append('./code')
from CaptionGenerator import CaptionGenerator

chainer.global_config.autotune = True
chainer.global_config.enable_backprop = False
chainer.global_config.type_check = False

#Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu",default=-1, type=int, help=u"GPU ID.CPU is -1")
parser.add_argument('--vocab',default='./data/MSCOCO/mscoco_caption_train2014_processed_dic.json', type=str,help='path to the vocaburary json')
parser.add_argument('--img',default='./sample_imgs/dog.jpg', type=str,help='path to the image')
parser.add_argument('--cnn-model', type=str, default='./data/ResNet50.model',help='place of the ResNet model')
parser.add_argument('--rnn-model', type=str, default='./data/caption_model.model',help='place of the caption model')
parser.add_argument('--beam',default=3, type=int,help='beam size in beam search')
parser.add_argument('--depth',default=50, type=int,help='depth limit in beam search')
parser.add_argument('--lang',default="<sos>", type=str,help='special word to indicate the langauge or just <sos>')
args = parser.parse_args()
tic = time.time()
caption_generator=CaptionGenerator(
    rnn_model_place=args.rnn_model,
    cnn_model_place=args.cnn_model,
    dictonary_place=args.vocab,
    beamsize=args.beam,
    depth_limit=args.depth,
    gpu_id=args.gpu,
    first_word= args.lang,
    )
toc = time.time()
print ("loading model time: ",toc-tic)
image_path =""
while True :
    image_path = input("image path (to exit type exit): ")
    if image_path == "exit": break
    tic = time.time()
    captions = caption_generator.generate(image_path)
    toc = time.time()
    print("caption_generator time: ",toc-tic)
    for caption in captions:
        print (" ".join(caption["sentence"]))
        print (caption["log_likelihood"])
    print()
