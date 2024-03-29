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
ambients_path = "/data2/adyotagupta/APS/190224_DCS/Data/Ambients/"
first_flag = True
first_im_min, first_im_max = None, None


def imsave(imgs, fname_base):
	def normalize_helper(image):
		global first_im_min, first_im_max, first_flag

		if first_flag:
        		first_im_min, first_im_max = np.median(image) - 3.0*image.std(),np.median(image) + 3.0*image.std() #np.amin(image), np.amax(image)
			first_flag = False
		else:
			# Clip image values
			image[image<first_im_min] = first_im_min
			image[image>first_im_max] = first_im_max

                #new_difference = 65535
                #imnew = (image - first_im_min)*(new_difference/(first_im_max - first_im_min))
		imnew = np.interp(image, (first_im_min, first_im_max), (0.,65535.))
                return imnew

	# We will normalize everything to the first frame
	first_max,first_min = np.amax(imgs[0]), np.amin(imgs[0])
	first_range = imgs[0].ptp()

	for i in np.arange(len(imgs)):
		fname = fname_base + '_' + str(i) + '.png'
		zgray = imgs[i]
		zgray = normalize_helper(zgray)
	#	zgray = (zgray-zgray.min())*first_range/zgray.ptp() + first_min
		with open(fname, 'wb') as f:
			writer = png.Writer(width=1024, height=1024, bitdepth=16, greyscale=True)
			zgray2list = zgray.tolist()
			writer.write(f, zgray2list)
	


with open(database,'rb') as fp:
	shot_list = pickle.load(fp)
shot_list = np.asarray(shot_list)



for o in np.arange(len(shot_list)):
	#Get shot from list and assemble the path where the registered deformation will be saved
	shot = shot_list[o]
	shot_path = ambients_path + shot.shot_num+'/'

	# Create the new directory, or overwrite it if it exists. Handles race conditions.
	if os.path.isdir(shot_path):
		shutil.rmtree(shot_path)
	try:
		os.makedirs(shot_path)
	except OSError as exc: #Handles Race Conditions
		if exc.errno != errno.EXIST:
			raise

	# Check if object has an unreliability condition. If not, we find the average reliable registration matrices
	print shot.shot_num,': '
	ambient_images = shot.get_normalized_ambient()
	imsave(ambient_images,shot_path+shot.shot_num)
	print "...saved !"
	#plotter_im(registered_deformation)
















#
