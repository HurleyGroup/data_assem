import numpy as np
import glob
import re
import copy
import cPickle as pickle
from itertools import izip_longest
from shot import shot


rootdir = "/data2/adyotagupta/APS/190224_DCS/Data/"
database = 'shot_database.pkl'
object_list = []


# Useful tools
shot_num_find = lambda d: d[-11:-3]
time = re.compile('[\d]{2}[_][\d]{2}[_][\d]{2}')
cam = re.compile('[C][\d]{1}')
frame = re.compile('[-][\d]{1}[.]')
sample = re.compile('sample_[\d]{1,2}')
recon = re.compile('rec[\d]{4}')

# Let's first traverse the DCS Data

# First find the shot numbers
dirs = np.asarray(glob.glob(rootdir+"LLNL_2019--02-22/*_C1"))
shot_nums = np.vectorize(shot_num_find)(dirs)

# Now let's do regex searches for each shot_num so we can find the relevant files
# and arrange them so that they are in the proper order when accessing them.
dirs = np.asarray(glob.glob(rootdir+"LLNL_2019--02-22/*"))
skip = False
print '\n\n'
for shot_num in shot_nums:
	# Obtains a list of files corresponding to each shot and checks whether a folder is 
	# empty or not -- if empty, then we just skip over the shot
	files = np.asarray([])
	for pwd in dirs[np.vectorize(re.search)(shot_num,dirs).astype(bool)]:
		new_files = np.asarray(glob.glob(pwd + '/*'))
		if new_files.size == 0:
			print 'WARNING: No images found in ' + pwd
			print 'Skipping shot...\n\n'
			skip = True
		files = np.hstack([files, new_files])
		
	if skip == True:
		skip = False
		continue

	# Extracts time stamps, camera numbers, and frame numbers through regex searches
	times = np.asarray(map(lambda x:x.group(), np.vectorize(time.search)(files)))
	cams = np.asarray(map(lambda x:x.group()[-1], np.vectorize(cam.search)(files)),dtype=int)
	frames = np.asarray(map(lambda x:x.group()[-2], np.vectorize(frame.search)(files)),dtype=int)

	# Separate by camera
	cam_1,cam_2,cam_3,cam_4 = files[cams==1],files[cams==2],files[cams==3],files[cams==4]

	# Sort for each camera, first by time-stamp, then by frame
	cam_1 = cam_1[np.lexsort((frames[cams==1],times[cams==1]))]
	cam_2 = cam_2[np.lexsort((frames[cams==2],times[cams==2]))]
	cam_3 = cam_3[np.lexsort((frames[cams==3],times[cams==3]))]
	cam_4 = cam_4[np.lexsort((frames[cams==4],times[cams==4]))]
	
	
	# Checks if there is an equal number of images present from each of the 4 cameras
	if not (cam_1.shape == cam_2.shape == cam_3.shape == cam_4.shape):
		print 'WARNING: Images collected from each camera are not of same dimensions for shot ' + shot_num
		print cam_1.shape, cam_2.shape, cam_3.shape, cam_4.shape

	
	# Arrange the images from each camera to obtain the final set of images in the correct order
	files = np.asarray([x for x in sum(izip_longest(cam_4[::-1],cam_3[::-1],cam_2[::-1],cam_1[::-1]), ()) if x is not None])
	files = files[::-1]

	
	# Create an object based on the shot number and transfer data to the object, and then to the list
	instance = shot(shot_num)
	instance.set_xpci(files)
	object_list.append(copy.deepcopy(instance))
		

# With the entire shot object set up, let's try to link the RECONSTRUCTED CT scan data
dirs = np.asarray(glob.glob(rootdir + "DSC_CT/*"))
dirs = dirs[np.asarray(np.vectorize(sample.search)(dirs),dtype=bool)] #removes tmp and other irrelevant directories
sample_nums_dirs = np.asarray(map(lambda x:x.group()[7:],np.vectorize(sample.search)(dirs)),dtype=int)


for o in np.arange(len(object_list)):
	sample_num = object_list[o].sample_num
	recon_files = np.asarray(glob.glob(str(dirs[sample_num == sample_nums_dirs][0] + "/sample_%d_Rec/*.tif"%sample_num)))
	recon_files = np.sort(recon_files[ np.asarray(np.vectorize(recon.search)(recon_files),dtype=bool)  ])
	object_list[o].set_ct(recon_files)
	object_list[o].registration()
	print object_list[o]

print '\n\nDatabase Created.\n'

# Now that we have linked all the data together, we can now save this "database" using cPickle to be retrieved and
# used at a later time
with open(database,'wb') as fp:
	pickle.dump(object_list,fp)

print 'Database saved. \n\n'






	











#	
