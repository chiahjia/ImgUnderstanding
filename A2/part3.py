from skimage import data
from skimage import io
from skimage import color
from scipy import ndimage
from matplotlib import pyplot as plt
import cv2
import numpy as np
from numpy import linalg as la
import math

path = "/Users/chiah/Documents/csc420/A2/UofT_lawn.jpg"
win_size = 5 #window size
sig = 10 #standard deviation
t = 40000 #threshold


'''
This function receives an image f(x,y) as input and returns its gradient g(x,y) magnitude as output using Sobel operation
'''
def gradMagnitude(gauss_img):

	rows= gauss_img.shape[0]
	cols = gauss_img.shape[1]

	sobel_x_filter = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
	sobel_y_filter  = [[-1, -2, -1], [0,0,0], [1,2,1]]

	img_grad = np.zeros([rows, cols])
	Ix_squared = np.zeros([rows, cols])
	Iy_squared = np.zeros([rows, cols])
	IxIy = np.zeros([rows, cols])
	pad_img = np.pad(gauss_img, pad_width=1, mode='constant', constant_values=0)

	#calculate the gradient

	#flip the filters
	flip_x_filter = np.zeros([3,3])
	flip_y_filter = np.zeros([3,3])

	for r in range(3):
		for c in range(3):
			flip_x_filter[2-r, 2-c] = sobel_x_filter[r][c]
			flip_y_filter[2-r, 2-c] = sobel_y_filter[r][c]

	temp_xmat = np.zeros([3,3])
	temp_ymat = np.zeros([3,3])

	#calculate the gradient
	for r in range (1, rows+1):
		for c in range (1, cols+1):
			#calculate the gradient using the x filter
			temp_xmat[0,0] = flip_x_filter[0,0] * pad_img[r-1, c-1]
			temp_xmat[0,1] = flip_x_filter[0,1] * pad_img[r-1, c]
			temp_xmat[0,2] = flip_x_filter[0,2] * pad_img[r-1, c+1]

			temp_xmat[1,0] = flip_x_filter[1,0] * pad_img[r, c-1]
			temp_xmat[1,1] = flip_x_filter[1,1] * pad_img[r, c]
			temp_xmat[1,2] = flip_x_filter[1,2] * pad_img[r, c+1]

			temp_xmat[2,0] = flip_x_filter[2,0] * pad_img[r+1, c-1]
			temp_xmat[2,1] = flip_x_filter[2,1] * pad_img[r+1, c]
			temp_xmat[2,2] = flip_x_filter[2,2] * pad_img[r+1, c+1]


			#calculate the gradient using the y filter
			temp_ymat[0,0] = flip_y_filter[0,0] * pad_img[r-1, c-1]
			temp_ymat[0,1] = flip_y_filter[0,1] * pad_img[r-1, c]
			temp_ymat[0,2] = flip_y_filter[0,2] * pad_img[r-1, c+1]

			temp_ymat[1,0] = flip_y_filter[1,0] * pad_img[r, c-1]
			temp_ymat[1,1] = flip_y_filter[1,1] * pad_img[r, c]
			temp_ymat[1,2] = flip_y_filter[1,2] * pad_img[r, c+1]

			temp_ymat[2,0] = flip_y_filter[2,0] * pad_img[r+1, c-1]
			temp_ymat[2,1] = flip_y_filter[2,1] * pad_img[r+1, c]
			temp_ymat[2,2] = flip_y_filter[2,2] * pad_img[r+1, c+1]

			IxIy[r-1, c-1] = np.sum(temp_xmat) * np.sum(temp_ymat)
			Ix_squared[r-1, c-1] = np.sum(temp_xmat)**2
			Iy_squared[r-1, c-1] = np.sum(temp_ymat)**2
			img_grad[r-1 ,c-1] = (np.sum(temp_xmat)**2 + np.sum(temp_ymat)**2)**(0.5)

	return img_grad, IxIy, Ix_squared, Iy_squared

def corner_detector(gray_img, rgb_img, window_size, stddev):

	img_grad, IxIy, Ix_squared, Iy_squared = gradMagnitude(gray_img)

	if window_size %2 == 0:
		window_size = window_size +1

	k = int((window_size-1)/2)
	window = np.zeros((window_size, window_size))
	window[k,k] = 1
	window =ndimage.gaussian_filter(window, sigma= stddev)


	M_11 = ndimage.correlate(Ix_squared, window, mode = 'nearest')
	M_22 = ndimage.correlate(Iy_squared, window, mode = 'nearest')
	M_12 = M_21 = ndimage.correlate(IxIy, window, mode = 'nearest')

	'''
	#uncomment to show scatterplot
	plt.plot(M_11, M_22, '.', color='blue')
	plt.xlabel('lambda1')
	plt.ylabel('lambda2')
	plt.title('lambda1 vs lambda2')
	plt.show()
	'''
	
	rows, cols = rgb_img.shape[0], rgb_img.shape[1]
	new_img = np.copy(rgb_img)

	#label the corners
	for r in range (rows):
		print("r is ", r)
		for c in range (cols):
			if M_11[r,c] > t and M_22[r,c] > t:
				new_img[r,c] = (255, 0, 0)

	plt.imshow(new_img)
	plt.show()




ori_img = cv2.imread(path)
rgb_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
gray_img = cv2.imread(path, 0)

corner_detector(gray_img, rgb_img, win_size, sig)






