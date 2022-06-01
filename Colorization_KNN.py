
"""

	Acknowledgement:
	The following codes are modified based on M. Cihat Ünal's Example-based-Image-Colorization-w-KNN
	from Github: https://github.com/ByUnal/Example-based-Image-Colorization-w-KNN.

"""

import argparse
import cv2
import glob
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pdb
from PIL import Image
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import skimage
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.color import rgb2lab, lab2rgb
from scipy import ndimage as nd
import time
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# function to read in the image in rgb and turn to lab
def convert_lab(filename):
	img = np.array(Image.open(filename).convert("RGB"))
	img_lab = rgb2lab(img) # Converting RGB to L*a*b
	L = img_lab[:,:,0]
	ab = img_lab[:,:,1:]
	return L, ab


# Feature Extraction
# This function extracts features based on gaussian/generic filter
def extract_all(img):

	img2 = img.reshape(-1)

	# First feature is grayvalue of pixel
	df = pd.DataFrame()
	df['GrayValue(I)'] = img2 

	# Second feature is GAUSSIAN filter with sigma=2
	gaussian_img = nd.gaussian_filter(img, sigma=2)
	gaussian_img1 = gaussian_img.reshape(-1)
	df['Gaussian s2'] = gaussian_img1

	# Third feature is GAUSSIAN fiter with sigma=4
	gaussian_img2 = nd.gaussian_filter(img, sigma=4)
	gaussian_img3 = gaussian_img2.reshape(-1)
	df['Gaussian s4'] = gaussian_img3

	# Third feature is generic filter that variance of pixel with size=3
	variance_img = nd.generic_filter(img, np.var, size=3)
	variance_img1 = variance_img.reshape(-1)
	df['Variance s3'] = variance_img1
    
	return df



# This function extracts average pixel value of neighboring pixels
# frame size : (distance * 2) + 1 x (distance * 2) + 1
def extract_neighbors_features(img, distance):

	height,width = img.shape
	X = []

	for x in range(height):
		for y in range(width):
			neighbors = []
			for k in range(x-distance, x+distance+1):
				for p in range(y-distance, y+distance+1):
					if x == k and p == y:
						continue
					elif ((k>0 and p>0 ) and (k<height and p<width)):
						neighbors.append(img[k][p])
					else:
						neighbors.append(0)
		
			X.append(sum(neighbors) / len(neighbors))

	return X


# This function extracts superpixels
# Every cell has a value in superpixel frame so 
# It is extracting superpixel value of every pixel
def superpixel(image,status):    
	if status:
		segments = slic(img_as_float(image), n_segments = 100, sigma = 5)
	else:
		segments = slic(img_as_float(image), n_segments = 100, sigma = 5,compactness=.1) 

	return segments


# # Function to save predicted images in Outputs folder in Dataset folder
def save_picture(num_val,y_predict,output_path):
	for i in range(num_val):
		img=y_predict[i*4096:(i+1)*4096,:].reshape(64,64,3)
		if output_path!=None:
			cv2.imwrite(output_path+'/val_'+str(i)+'.JPEG',img)
		else:
			print("Invalid output path!")

# save Lab image as RGB
def save_as_rgb(L, ab, output_path):

	Lab = np.concatenate((L[:,0].reshape((len(L),1)), ab), axis=-1)
	if output_path==None:
		rgb_imgs = []
	for i in range(Lab.shape[0]//4096):
		img_rgb = lab2rgb(Lab[i*4096:(i+1)*4096,:].reshape((64,64,3)).astype(np.float64))
		if output_path==None:
			rgb_imgs.append(img_rgb.reshape((4096,3)))
		else:
			plt.imsave(output_path+'/val_'+str(i)+'.JPEG',img_rgb)

	if output_path==None:
		rgb_imgs=np.stack(rgb_imgs, axis=0)
	else:
		rgb_imgs=None

	return rgb_imgs

def main():

	distance=4

	if not args.load_processed:

		if args.if_rgb:# RGB channels target
			
			# groundtruth rgb data for training #'/Users/hanqingyao/Desktop/cs 231n project/tiny-imagenet-200/train_data
			train_rgb = [cv2.imread(file, 1) for file in glob.glob(args.train_rgb_path+'/*.JPEG')]
			# gray scale data for training
			train_gray = [cv2.imread(file, 0) for file in glob.glob(args.train_gray_path+'/*.JPEG')]
			# groundtruth rgb data for validation
			val_rgb = [cv2.imread(file, 1) for file in glob.glob(args.val_rgb_path+'/*.JPEG')]
			# gray scale data for validation
			val_gray = [cv2.imread(file, 0) for file in glob.glob(args.val_gray_path+'/*.JPEG')]

			if args.load_test:
				# groundtruth rgb data for testing
				test_rgb = [cv2.imread(file, 1) for file in glob.glob(args.test_rgb_path+'/*.JPEG')]
				# gray scale data for testing
				test_gray = [cv2.imread(file, 0) for file in glob.glob(args.test_gray_path+'/*.JPEG')]
		else: # Lab channels target

			# gray scale and ab channel labels for training
			train_gray, y_train_ab= [], []
			for file in glob.glob(args.train_rgb_path+'/*.JPEG'):
				L, ab = convert_lab(file)
				train_gray.append(L)
				y_train_ab.append(ab)

			# gray scale and ab channel labels for validation
			val_gray = []
			for file in glob.glob(args.val_rgb_path+'*.JPEG'):
				L, _ = convert_lab(file)
				val_gray.append(L)

			if args.load_test:
				# gray scale and ab channel labels for testing
				test_gray= []
				for file in glob.glob(args.test_rgb_path+'/*.JPEG'):
					L, ab = convert_lab(file)
					test_gray.append(L)

		# Processing grayscale data to extract features
		if args.if_rgb:
			orig_data_dict={"X_train": train_gray,"X_val":val_gray,"y_train":train_rgb,"y_val":val_rgb}
		else:
			orig_data_dict={"X_train": train_gray,"X_val":val_gray,"y_train_ab":y_train_ab}

		if args.load_test:
			orig_data_dict["X_test"]=test_gray
			if args.if_rgb:
				orig_data_dict["y_test"]=test_rgb

		data_dict={k:[] for k in orig_data_dict.keys()}
		for key, value in orig_data_dict.items():
			print(key)
			if key.find('y_')!=-1:
				# preparing y (b, g, r)
				for i in range(len(value)):
					print(key," on y ",i)
					if args.if_rgb:
						y = value[i].reshape((-1,3))
					else:
						y = value[i].reshape((-1,2))
					data_dict[key].append(y)
				np.save(args.processed_data_path+key+"_rgb.npy" if args.if_rgb else "_ab.npy",np.vstack(data_dict[key]))
			else:
				# preparing X variable
				for i in range(len(value)):
					print(key," on x ",i)
					print("extract on {} {}".format(key,i))
					X1 = extract_all(value[i]).values
					X2 = superpixel(value[i],False).reshape(-1,1)
					X3 = extract_neighbors_features(value[i],distance)
					X = np.c_[X1, X2, X3]
					data_dict[key].append(X)
				np.save(args.processed_data_path+key+"_rgb.npy" if args.if_rgb else "_ab.npy",np.vstack(data_dict[key]))

	else:
		# load preprocessed data
		if args.if_rgb:
			data_dict=dict()
			data_dict['X_train'] = np.load(args.processed_data_path+'/X_train_rgb.npy')
			data_dict['X_val'] = np.load(args.processed_data_path+'/X_val_rgb.npy')
			data_dict['y_train'] = np.load(args.processed_data_path+'/y_train_rgb.npy')
			data_dict['y_val'] = np.load(args.processed_data_path+'/y_val_rgb.npy')
		else:
			data_dict=dict()
			data_dict['X_train'] = np.load(args.processed_data_path+'/X_train_ab.npy')
			data_dict['X_val'] = np.load(args.processed_data_path+'/X_val_ab.npy')
			data_dict['y_train_ab'] = np.load(args.processed_data_path+'/y_train_ab.npy')
			data_dict['y_train'] = np.load(args.processed_data_path+'/y_train_rgb.npy')
			data_dict['y_val'] = np.load(args.processed_data_path+'/y_val_rgb.npy')

	# Set up KNN model
	if args.if_rgb:
		knn_clf = KNeighborsClassifier(n_neighbors=args.k)
	else:
		knn_clf = KNeighborsRegressor(n_neighbors=args.k)


	if args.if_rgb:
		start_time = time.time()
		knn_clf.fit(data_dict['X_train'][:args.train_size*4096,:],data_dict['y_train'][:args.train_size*4096,:])
		y_val_pred=knn_clf.predict(data_dict['X_val'][:args.val_size*4096,:])
		error = np.mean(np.abs(np.vstack(data_dict['y_val'][:args.val_size,]).astype('float')-y_val_pred.astype('float')),axis=0)
		print("Validation MAE",np.mean(error))
		y_train_pred=knn_clf.predict(data_dict['X_train'][:args.train_size*4096,:])
		error = np.mean(np.abs(data_dict['y_train'][:args.train_size*4096,].astype('float')-y_train_pred.astype('float')),axis=0)
		print("Training MAE",np.mean(error))
		print("Total time is {}".format(time.time()-start_time))
		save_picture(args.val_size,y_val_pred,args.output_path)
	else:
		start_time = time.time()
		knn_clf.fit(data_dict['X_train'][:args.train_size*4096,:],data_dict['y_train_ab'][:args.train_size*4096,:])
		y_val_pred=knn_clf.predict(data_dict['X_val'][:args.val_size*4096,:])
		lab_val_pred = save_as_rgb(data_dict['X_val'][:args.val_size*4096,:],y_val_pred,None)
		error = np.mean(np.abs(np.vstack(data_dict['y_val'][:args.val_size,]).astype('float')-np.vstack(lab_val_pred)*255),axis=0)
		print("Validation MAE",np.mean(error))
		y_train_pred=knn_clf.predict(data_dict['X_train'][:args.train_size*4096,:])
		lab_train_pred = save_as_rgb(data_dict['X_train'][:args.train_size*4096,:],y_train_pred,None)
		error = np.mean(np.abs(data_dict['y_train'][:args.train_size*4096,].astype('float')-np.vstack(lab_train_pred)*255),axis=0)
		print("Training MAE",np.mean(error))
		print("Total time is {}".format(time.time()-start_time))
		save_as_rgb(data_dict['X_val'][:args.val_size*4096,:],y_val_pred,args.output_path)





if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--train_rgb_path", help="Path to RGB images of training data", type=str)
	parser.add_argument("--val_rgb_path", help="Path to RGB images of validation data", type=str)
	parser.add_argument("--test_rgb_path", help="Path to RGB images of testing data", type=str)
	parser.add_argument("--train_gray_path", help="Path to grayscale images of training data", type=str)
	parser.add_argument("--val_gray_path", help="Path to grayscale images of validation data", type=str)
	parser.add_argument("--test_gray_path", help="Path to grayscale images of testing data", type=str)
	parser.add_argument("--processed_data_path", help="Output path to save feature extraction processed data", type=str)
	parser.add_argument("--load_test", help="Indicator for whether to load test data", type=str, default=0)
	parser.add_argument("--load_processed", help="Indicator for whether to load feature etracted data", type=int, default=0)
	parser.add_argument("--train_size", help="Number of training images to use", type=int)
	parser.add_argument("--val_size", help="Number of validation images to use", type=int)
	parser.add_argument("--distance", help="Distance used as to select neighboring pixels for calculating average pixel value", \
		type=int, default=4)
	parser.add_argument("--k", help="Number of neighbors in KNN model", type=int, default=3)
	parser.add_argument("--output_path", help="Output path to save predicted images", type=str)
	parser.add_argument("--if_rgb", help="Indicator for whether to use RGB channel or Lab channel", type=int, default=1)
	args = parser.parse_args()
	print(args)
	main()


