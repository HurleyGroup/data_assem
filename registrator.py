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
	shot_path = registered_deformations_path + shot.shot_num+'/'

	if shot.shot_num != '19-4-039': continue

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

	print shot.shot_num,': ', deformation.shape, registration_info.shape

	registered_deformation = []

	for i in np.arange(len(deformation)):
		if (i%4) != 0:
			if shot.shot_num == '19-4-025':
				shot194026 = (shot_list[np.asarray([x.shot_num for x in shot_list]) == '19-4-026'])[0]
				img = cv2.warpAffine(deformation[i], shot194026.registration_matrices[i%4], deformation[i].shape, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)


			elif shot.shot_num == '19-4-034' or shot.shot_num=='19-4-039':
				shot194036 = (shot_list[np.asarray([x.shot_num for x in shot_list]) == '19-4-036'])[0]
				img = cv2.warpAffine(deformation[i], shot194036.registration_matrices[i%4], deformation[i].shape, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)


			elif shot.shot_num == '19-4-039':
				shot194038 = (shot_list[np.asarray([x.shot_num for x in shot_list]) == '19-4-040'])[0]
				img = cv2.warpPerspective(deformation[i],shot194038.registration_matrices[i%4],deformation[i].shape)


			elif shot.shot_num == '19-4-036' or shot.shot_num == '19-4-028' or shot.shot_num == '19-4-027' or \
				 shot.shot_num == '19-4-032' or shot.unreliable_registration:
				img = cv2.warpAffine(deformation[i], registration_info[i%4], deformation[i].shape, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
			else:
				img = cv2.warpPerspective(deformation[i],registration_info[i%4],deformation[i].shape)
	
		else:
			img = deformation[i]

		registered_deformation.append(img)



	registered_deformation = np.asarray(registered_deformation)
	imsave(registered_deformation,shot_path+shot.shot_num)
	print shot.shot_num,': ', registered_deformation.shape
	#plotter_im(registered_deformation)
















#
