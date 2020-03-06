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

import argparse
import numpy as np
import numpy_indexed as npi
import math
from chainer import cuda
import chainer.functions as F
from chainer import cuda, Function, gradient_check, Variable, optimizers
from chainer import serializers
from translator import translate_eg_ar
sys.path.append('./code')
from CaptionGenerator import CaptionGenerator

chainer.global_config.autotune = True
chainer.global_config.enable_backprop = False
chainer.global_config.type_check = False

class image_captioning ():
    def __init__(self,
    	rnn_model_place='data/caption_en_model40.model',
    	cnn_model_place='data/ResNet50.model',
    	dictonary_place='data/MSCOCO/mscoco_caption_train2014_processed_dic.json',
    	beamsize=3,
    	depth_limit=50,
    	gpu_id=-1,
    	first_word="<sos>"):

    	self.caption_generator=CaptionGenerator(
	    	rnn_model_place=rnn_model_place,
	    	cnn_model_place=cnn_model_place,
	    	dictonary_place=dictonary_place,
	    	beamsize=beamsize,
	    	depth_limit=depth_limit,
	    	gpu_id=gpu_id,
	    	first_word= first_word,
    	)

    def generate_from_img_nparray(self,image_array):
        image_feature=self.caption_generator.cnn_model(image_array, "feature").data.reshape(1,1,2048)
        return self.caption_generator.generate_from_img_feature(image_feature)


if __name__ == '__main__':
    image_captioning=image_captioning()
    np_image = image_captioning.caption_generator.image_loader.load("test7.jpg")
    captions = image_captioning.generate_from_img_nparray(np_image)
    for caption in captions:
  #     print (" ".join(caption["sentence"]))
       print (caption["sentence"])
       np_caption=np.array(caption["sentence"])
       delete_words=np.array(["<sos>", "<eos>", 'a', 'an'])
       np_caption=np_caption[(np_caption!='<eos>') & (np_caption !='<sos>') & (np_caption!= 'a') & (np_caption!='an')]
       #prepare input for translate_eg_ar
       caption_list=list(np_caption)
       ar=translate_eg_ar(caption_list)
       translated_sen=" ".join(ar)
       print(translated_sen)
       print (caption["log_likelihood"])


