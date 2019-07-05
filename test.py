import numpy as np
import cPickle as pickle
from plotting import plotter, plotter_im
from matplotlib import pyplot as plt
import cv2
import shutil
import os
import png
import sys

shot_list = None
database = 'shot_database.pkl'

with open(database,'rb') as fp:
        shot_list = pickle.load(fp)
shot_list = np.asarray(shot_list)

sample_1 = shot_list[np.asarray(map(lambda o: o.sample_num, shot_list))==21][0]
sample_2 = shot_list[np.asarray(map(lambda o: o.sample_num, shot_list))==2][0]


amb = sample_1.get_ambient_avg(2)
br = sample_1.get_bright_avg(2)
dk = sample_1.get_dark_avg(2)
im = (amb-dk)/(br-dk)
im_old = sample_1.get_normalized_ambient(cam=2)

print im.max(), im.min(), im.std()
print dk.max(), dk.min(), dk.std()
print br.max(), br.min(), br.std()
print amb.max(), amb.min(), amb.std()

plt.figure()
plt.imshow(amb)
plt.figure()
plt.imshow(br)
plt.figure()
plt.imshow(dk)
plt.figure()
plt.imshow(im)
plt.figure()
plt.imshow(im_old)
plt.show()

