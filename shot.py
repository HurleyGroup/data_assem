import numpy as np
from matplotlib import pyplot as plt
import cPickle as pickle
import re
from itertools import izip_longest
import cv2

class shot:

	def __init__(self, shot_num):
		self.shot_num = shot_num
		self.proj_mass = None
		self.powder_load = None
		self.primer_load = None
		self.proj_diam = None
		self.chamb_vac = None
		self.breech_pres = None
		self.velocity = None
		self.sample_num = None
		self.xpci = None
		self.ct = None
		self.X2 = None
		self.Y = None
		self.gap = None
		self.bright = None
		self.dark = None
		self.ambient = None
		self.fsingle = None
		self.cpumps = None
		self.registration_matrices = None
		self.unreliable_registration = False
		self.__read_chart(self.shot_num) # Populates each instance with information from worksheet + other hand-inputted stuff from the CSV file		

	
	def __repr__(self):
		print '\nShot ', self.shot_num, ': \n'
		print 'self.proj_mass = ', self.proj_mass
		print 'self.powder_load = ', self.powder_load
		print 'self.primer_load = ', self.primer_load
		print 'self.proj_diam = ', self.proj_diam
		print 'self.chamb_vac = ', self.chamb_vac
		print 'self.breech_pres = ', self.breech_pres
		print 'self.velocity = ', self.velocity
		print 'self.sample_num = ', self.sample_num
		print 'self.xpci.shape = ', self.xpci.shape
		print 'self.ct.shape = ', self.ct.shape
		print 'self.X2, self.Y = ', self.X2, self.Y
		print 'self.gap = ', self.gap
		print 'self.bright, self.dark, self.ambient = ', self.bright, self.dark, self.ambient
		print 'self.fsingle, self.cpumps = ', self.fsingle, self.cpumps
		print 'self.registration_matrices.shape = ', self.registration_matrices.shape
		print 'self.unreliable_registration = ', self.unreliable_registration
		return '\n'




	def __read_chart(self,shot_num):
		chart = np.genfromtxt("shot_wkshts.csv",dtype=str,delimiter=',',skip_header=1)
		chart = chart[chart[:,0]==shot_num][0]
		self.sample_num,self.proj_mass,self.powder_load,self.primer_load = int(chart[1]), float(chart[2]), float(chart[3]), float(chart[4])
		self.proj_diam,self.chamb_vac,self.breech_pres,self.velocity = self.__process_diams(chart[5]),float(chart[6]), float(chart[7]), float(chart[8])
		self.X2, self.Y, self.gap, self.bright = float(chart[9]), float(chart[10]), int(chart[11]), tuple([int(chart[12])//100, int(chart[12])%100])
		self.dark, self.ambient, self.fsingle, self.cpumps = tuple([int(chart[13])//100, int(chart[13])%100]), tuple([int(chart[14])//100, int(chart[14])%100]), int(chart[15]), int(chart[16])
		self.unreliable_registration = bool(int(chart[17]))

	def __process_diams(self,measurements):
		finder = re.compile("[.][\d]{5}")
		if len(measurements) <= 6:
			return float(measurements)
		else:
			return np.mean(np.asarray(finder.findall(measurements),dtype=np.float64))

	# Returns all relevant paths to XPCI images, sorted, relevant to given shot
	def set_xpci(self, arr):
		assert arr.dtype.char == 'S' # Check if arr is an array of strings, which should be the list of files
		self.xpci = arr[:]

	
	# Returns all relevant paths to reconstructed CT images, sorted, relevant to given shot
	def set_ct(self, arr):
		assert arr.dtype.char == 'S'
		self.ct = arr[:]

	
	# Returns all relevant paths for bright-field images, sorted, relevant to given shot
	# If cam == 0, then we get all of the brights by default. We can specify the camera
	# to get those specific bright-field images
	def get_brights(self, cam=0):
		assert (cam>=0) and (cam <=4)
		re_index = re.compile("[\d]{2}-F")
		re_cam = re.compile("[C][\d]{1}")
		
		retrieved = np.asarray(map(lambda x:x.group()[:-2], np.vectorize(re_index.search)(self.xpci)), dtype=int)
		cams = np.asarray(map(lambda x:x.group()[1], np.vectorize(re_cam.search)(self.xpci) ), dtype=int )

		mask = (retrieved >= self.bright[0]) & (retrieved <= self.bright[1])
		cam_mask = cams==cam
		
		if cam > 0:
			mask = mask & cam_mask

		return self.xpci[mask]

	# Returns an average of bright-field images for the corresponding camera
	def get_bright_avg(self, camera):
		assert (camera>=1) and (camera<=4)
		paths = self.get_brights(cam=camera)
		bright_imgs = []
		for p in paths:
			bright_imgs.append(plt.imread(p))
		bright_imgs = np.asarray(bright_imgs)
		return np.mean(bright_imgs, axis=0)

		

	# Returns all relevant paths for dark-field images, sorted, relevant to given shot	
	# If cam == 0, then we get all of the brights by default. We can specify the camera
	# to get those specific bright-field images
	def get_darks(self, cam=0):
		assert (cam>=0) and (cam <=4)
		re_index = re.compile("[\d]{2}-F")
		re_cam = re.compile("[C][\d]{1}")

		retrieved = np.asarray(map(lambda x:x.group()[:-2], np.vectorize(re_index.search)(self.xpci)), dtype=int)
		cams = np.asarray(map(lambda x:x.group()[1], np.vectorize(re_cam.search)(self.xpci) ), dtype=int )

		mask = (retrieved >= self.dark[0]) & (retrieved <= self.dark[1])
		cam_mask = cams==cam

		if cam > 0:
			mask = mask & cam_mask

		return self.xpci[mask]


	# Returns an average of dark-field images for the corresponding camera
	def get_dark_avg(self, camera):
		assert (camera>=1) and (camera<=4)
		paths = self.get_darks(cam=camera)
		dark_imgs = []
		for p in paths:
			dark_imgs.append(plt.imread(p))
		dark_imgs = np.asarray(dark_imgs)
		return np.mean(dark_imgs, axis=0)


	# Returns all relevant paths for images during the onset of impact, of either all the cameras
	# or the relevant camera specified
	def get_impact(self, cam=0):
		assert (cam>=0) and (cam<=4)
		re_index = re.compile("[\d]{2}-F")				
		re_cam = re.compile("[C][\d]{1}")


		retrieved = np.asarray(map(lambda x:x.group()[:-2], np.vectorize(re_index.search)(self.xpci)), dtype=int)
		cams = np.asarray(map(lambda x:x.group()[1], np.vectorize(re_cam.search)(self.xpci) ), dtype=int )

		mask = retrieved >= self.cpumps
		cam_mask = cams==cam

		if cam > 0:
			mask = mask & cam_mask

		return self.xpci[mask]


	# Returns a series of normalized images of the impact using the bright and dark field images
	def get_normalized_impact(self):
		def normalize_helper(image, dark, bright):
			im_min, im_max = np.amin(image), np.amax(image)
			new_difference = bright-dark
			return np.multiply(image - im_min,new_difference/(im_max - im_min)) + dark

		impact_images_cam_1 = np.asarray(map(lambda x: plt.imread(x), self.get_impact(cam=1)))
		impact_images_cam_2 = np.asarray(map(lambda x: plt.imread(x), self.get_impact(cam=2)))
		impact_images_cam_3 = np.asarray(map(lambda x: plt.imread(x), self.get_impact(cam=3)))
		impact_images_cam_4 = np.asarray(map(lambda x: plt.imread(x), self.get_impact(cam=4)))
	
		impact_images_cam_1 = np.asarray(map(lambda x: normalize_helper(x,self.get_dark_avg(camera=1),self.get_bright_avg(camera=1)), impact_images_cam_1))	
		impact_images_cam_2 = np.asarray(map(lambda x: normalize_helper(x,self.get_dark_avg(camera=2),self.get_bright_avg(camera=2)), impact_images_cam_2))	
		impact_images_cam_3 = np.asarray(map(lambda x: normalize_helper(x,self.get_dark_avg(camera=3),self.get_bright_avg(camera=3)), impact_images_cam_3))	
		impact_images_cam_4 = np.asarray(map(lambda x: normalize_helper(x,self.get_dark_avg(camera=4),self.get_bright_avg(camera=4)), impact_images_cam_4))	
		
		normalized_impact = np.asarray([x for x in sum(izip_longest(impact_images_cam_1,impact_images_cam_2,impact_images_cam_3,impact_images_cam_4), ()) if x is not None])

		return normalized_impact 


	# Returns a normalized ambient image of the sample using the bright and dark field images
	# Returns the normalized image of the relevant camera, if camera is specified
	def get_normalized_ambient(self,cam=0):
		assert (cam>=0) and (cam<=4)
		def normalize_helper(image, dark, bright):
			im_min, im_max = np.amin(image), np.amax(image)
			new_difference = bright-dark
			return np.multiply(image - im_min,new_difference/(im_max - im_min)) + dark

		ambient_image_cam_1 = self.get_ambient_avg(camera=1)
		ambient_image_cam_2 = self.get_ambient_avg(camera=2)
		ambient_image_cam_3 = self.get_ambient_avg(camera=3)
		ambient_image_cam_4 = self.get_ambient_avg(camera=4)
	
		ambient_image_cam_1 = normalize_helper(ambient_image_cam_1,self.get_dark_avg(camera=1),self.get_bright_avg(camera=1))	
		ambient_image_cam_2 = normalize_helper(ambient_image_cam_2,self.get_dark_avg(camera=2),self.get_bright_avg(camera=2))	
		ambient_image_cam_3 = normalize_helper(ambient_image_cam_3,self.get_dark_avg(camera=3),self.get_bright_avg(camera=3))	
		ambient_image_cam_4 = normalize_helper(ambient_image_cam_4,self.get_dark_avg(camera=4),self.get_bright_avg(camera=4))	
		
		normalized_ambient = np.asarray([ambient_image_cam_1, ambient_image_cam_2, ambient_image_cam_3, ambient_image_cam_4])

		if cam > 0:
			return normalized_ambient[cam-1]

		return normalized_ambient



	# Returns all relevant paths for images that contain the ambient shot of the sample, of either all the cameras
	# or the relevant camera specified
	def get_ambient(self, cam=0,):
		assert (cam>=0) and (cam<=4)
		re_index = re.compile("[\d]{2}-F")				
		re_cam = re.compile("[C][\d]{1}")

		retrieved = np.asarray(map(lambda x:x.group()[:-2], np.vectorize(re_index.search)(self.xpci)), dtype=int)
		cams = np.asarray(map(lambda x:x.group()[1], np.vectorize(re_cam.search)(self.xpci) ), dtype=int )

		mask = (retrieved >= self.ambient[0]) & (retrieved <= self.ambient[1])
		cam_mask = cams==cam

		if cam > 0:
			mask = mask & cam_mask

		return self.xpci[mask]


	# Returns an average of ambient images for the corresponding camera
	def get_ambient_avg(self, camera):
		assert (camera>=1) and (camera<=4)
		paths = self.get_ambient(cam=camera)
		ambient_imgs = []
		for p in paths:
			ambient_imgs.append(plt.imread(p))
		ambient_imgs = np.asarray(ambient_imgs)
		return np.mean(ambient_imgs, axis=0)




	def flip_reliability(self):
		self.unreliable = not self.unreliable



	# Returns the registered images, and the inverse of the "deformation gradient" 
	# This serves as a helper method to the registration method below.
	# Here, X = h*x. So the true deformation gradient is inverse of h
	def __alignImages(self, im1, im2, MAX_FEATURES=1000, GOOD_MATCH_PERCENT=0.15):
 
		# Convert images to grayscale
	  	im1Gray = im1 #cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
	  	im2Gray = im2 #cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
   
	 	# Detect ORB features and compute descriptors.
 		orb = cv2.ORB_create(MAX_FEATURES)
 		keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
 		keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
   
		# Match features.
		matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
		matches = matcher.match(descriptors1, descriptors2, None)
  
		# Sort matches by score
		matches.sort(key=lambda x: x.distance, reverse=False)

		# Remove not so good matches
		numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
		matches = matches[:numGoodMatches]

		# Extract location of good matches
		points1 = np.zeros((len(matches), 2), dtype=np.float32)
		points2 = np.zeros((len(matches), 2), dtype=np.float32)
 
		for i, match in enumerate(matches):
			points1[i, :] = keypoints1[match.queryIdx].pt
			points2[i, :] = keypoints2[match.trainIdx].pt
   
		# Find homography
		h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
 
   
		return h


	def __alignImages_ECC(self,im1,im2,model_type=cv2.MOTION_AFFINE, number_of_iterations=100):
		# Convert images to grayscale
		im1_gray = im1 #cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
		im2_gray = im2 #cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
 
		# Find size of image1
		sz = im1.shape
 
		# Define the motion model
		warp_mode = model_type
 
		# Define 2x3 or 3x3 matrices and initialize the matrix to identity
		if warp_mode == cv2.MOTION_HOMOGRAPHY :
			warp_matrix = np.eye(3, 3, dtype=np.float32)
		else :
			warp_matrix = np.eye(2, 3, dtype=np.float32)
 
		# Specify the number of iterations.
		# number_of_iterations = 100;#5000
 
		# Specify the threshold of the increment
		# in the correlation coefficient between two iterations
		termination_eps = 1e-10;
 
		# Define termination criteria
		criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
 
		# Run the ECC algorithm. The results are stored in warp_matrix.
		(cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)
 
		"""
		if warp_mode == cv2.MOTION_HOMOGRAPHY :
			# Use warpPerspective for Homography 
			im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
		else :
			# Use warpAffine for Translation, Euclidean and Affine
			im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
 		"""

		return warp_matrix


	# Returns the image registration of cameras 1, 2, 3, and 4, with respect to camera 1.
	def registration(self,cam=0):
		#assert self.unreliable_registration == False # Ensure that the registration that we obtained with the feature extraction method is reliable
		def normalize_helper(image):
                        im_min, im_max = np.amin(image), np.amax(image)
                        new_difference = 255
                        return (image - im_min)*(new_difference/(im_max - im_min))
		

		im1 = np.asarray(normalize_helper(self.get_normalized_ambient(cam=1)),dtype=np.uint8)	
		im2 = np.asarray(normalize_helper(self.get_normalized_ambient(cam=2)),dtype=np.uint8)
		im3 = np.asarray(normalize_helper(self.get_normalized_ambient(cam=3)),dtype=np.uint8)
		im4 = np.asarray(normalize_helper(self.get_normalized_ambient(cam=4)),dtype=np.uint8)
		
		if (self.shot_num == '19-4-025') or (self.shot_num == '19-4-032'):
			regis_1 = self.__alignImages_ECC(im1,im1,model_type=cv2.MOTION_EUCLIDEAN)
			regis_2 = self.__alignImages_ECC(im1,im2,model_type=cv2.MOTION_EUCLIDEAN)
			regis_3 = self.__alignImages_ECC(im1,im3,model_type=cv2.MOTION_EUCLIDEAN)
			regis_4 = self.__alignImages_ECC(im1,im4,model_type=cv2.MOTION_EUCLIDEAN)

		elif (self.shot_num == '19-4-036') or (self.shot_num == '19-4-028') or self.unreliable_registration:
			if self.shot_num == '19-4-027' or self.shot_num == '19-4-039':
				regis_1 = self.__alignImages_ECC(im1,im1,model_type=cv2.MOTION_EUCLIDEAN,number_of_iterations=5000)
				regis_2 = self.__alignImages_ECC(im1,im2,model_type=cv2.MOTION_EUCLIDEAN,number_of_iterations=5000)
				regis_3 = self.__alignImages_ECC(im1,im3,model_type=cv2.MOTION_EUCLIDEAN,number_of_iterations=5000)
				regis_4 = self.__alignImages_ECC(im1,im4,model_type=cv2.MOTION_EUCLIDEAN,number_of_iterations=5000)
			else:
				regis_1 = self.__alignImages_ECC(im1,im1)
				regis_2 = self.__alignImages_ECC(im1,im2)
				regis_3 = self.__alignImages_ECC(im1,im3)
				regis_4 = self.__alignImages_ECC(im1,im4)

		else:
			regis_1 = self.__alignImages(im1,im1)
			regis_2 = self.__alignImages(im2,im1)
			regis_3 = self.__alignImages(im3,im1)
			regis_4 = self.__alignImages(im4,im1)


		regis = np.asarray([regis_1,regis_2,regis_3,regis_4] )
		
		if cam>0 :
			return regis[cam-1]
		
		self.registration_matrices = regis[:]

		return regis









#
