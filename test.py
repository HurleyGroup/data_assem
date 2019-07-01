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
registered_deformations_path = "/data2/adyotagupta/APS/190224_DCS/Data/Registered_Impact/"

def imsave(imgs, fname_base):
	# We will normalize everything to the first frame
	first_max,first_min = np.amax(imgs[0]), np.amin(imgs[0])
	first_range = imgs[0].ptp()

	for i in np.arange(len(imgs)):
		fname = fname_base + '_' + str(i)
		zgray = imgs[i]
		zgray = (zgray-zgray.min())*first_range/zgray.ptp() + first_min
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
	shot_path = registered_deformations_path + shot.shot_num+'/'

	# Create the new directory, or overwrite it if it exists. Handles race conditions.
	if os.path.isdir(shot_path):
		shutil.rmtree(shot_path)
	try:
		os.makedirs(shot_path)
	except OSError as exc: #Handles Race Conditions
		if exc.errno != errno.EXIST:
			raise

	# Check if object has an unreliability condition. If not, we find the average reliable registration matrices
	deformation = shot.get_normalized_impact()
	registration_info = shot.registration_matrices

	registered_deformation = []
	for i in np.arange(len(deformation)):
		if shot.shot_num == '19-4-036' or shot.shot_num == '19-4-028' or shot.shot_num == '19-4-027' or \
			 shot.shot_num == '19-4-032' or shot.shot_num == '19-4-025' or shot.unreliable_registration:

			img = cv2.warpAffine(deformation[i], registration_info[i%4], deformation[i].shape, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

		else:
			img = cv2.warpPerspective(deformation[i],registration_info[i%4],deformation[i].shape)
	
		registered_deformation.append(img)



	registered_deformation = np.asarray(registered_deformation)
	imsave(registered_deformation,shot_path+shot.shot_num)
	print shot.shot_num,': ', registered_deformation.shape
	#plotter_im(registered_deformation)
















#
