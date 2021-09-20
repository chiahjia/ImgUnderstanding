# ASSIGNMENT 1 PART 2 QUESTION 4
from skimage import data
from skimage import io
from skimage import color

from matplotlib import pyplot as plt
import cv2
import numpy as np
from numpy import linalg as la
import math

path = '/Users/chiah/Documents/csc420/A1/Q6.png'

'''
------------------------------------ Step 1 ------------------------------------
This function computes and returns the 𝚔𝚜𝚒𝚣𝚎×1 matrix of Gaussian filter coefficients

ksize: Aperture size. It should be odd and positive.
sigma: Gaussian standard deviation. 
	   If it is non-positive, it is computed from ksize as sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8

Note: the computation for neative sigma is based on the documentation provided by opencv for getGaussianKernel()
source: https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gac05a120c1ae92a6060dd0db190a61afa

'''
def getGaussKernel(x_size, y_size, sigma):

	# check ksize
	if (x_size % 2 == 0) or (x_size < 0) or (y_size % 2 == 0) or (y_size < 0):
		print("USAGE: getGaussKernel(x_size, y_size, sigma)")
		print("x_size and y_size must be odd and positive")
		exit()

	if (sigma < 0):
		sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8

	coefficient = np.zeros([x_size, y_size])
	x_centre = math.floor(x_size / 2)
	y_centre = math.floor(y_size / 2)

	left_term = 1 / (2 * math.pi * sigma) ** 2

	for x in range(x_size):
		for y in range(y_size):
			exponent = -((x - x_centre)**2 + (y - y_centre)**2) / sigma**2;
			coefficient[x, y] = left_term * math.exp(exponent)

	return coefficient 

#function to apply the gaussian filter on the input image
def applyGauss (coefficient, img_path):
	gray_img = cv2.imread(img_path, 0) #read the input image as grayscale
	
	rows= gray_img.shape[0]
	cols = gray_img.shape[1]
	pad_img = np.pad(gray_img, pad_width=1, mode='constant', constant_values=0)

	img_gauss = np.zeros([rows, cols])

	temp_mat = np.zeros([3,3])
	for r in range(rows):
		for c in range(cols):
			temp_mat[0,0] = coefficient[0,0] * pad_img[r-1, c-1]
			temp_mat[0,1] = coefficient[0,1] * pad_img[r-1, c]
			temp_mat[0,2] = coefficient[0,2] * pad_img[r-1, c+1]

			temp_mat[1,0] = coefficient[1,0] * pad_img[r, c-1]
			temp_mat[1,1] = coefficient[1,1] * pad_img[r, c]
			temp_mat[1,2] = coefficient[1,2] * pad_img[r, c+1]

			temp_mat[2,0] = coefficient[2,0] * pad_img[r+1, c-1]
			temp_mat[2,1] = coefficient[2,1] * pad_img[r+1, c]
			temp_mat[2,2] = coefficient[2,2] * pad_img[r+1, c+1]


			img_gauss[r,c] = np.sum(temp_mat)

	return img_gauss


'''
------------------------------------ Step 2 ------------------------------------
This function receives an image f(x,y) as input and returns its gradient g(x,y) magnitude as output using Sobel operation
'''
def gradMagnitude(gauss_img):

	rows= gauss_img.shape[0]
	cols = gauss_img.shape[1]

	sobel_x_filter = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
	sobel_y_filter  = [[-1, -2, -1], [0,0,0], [1,2,1]]

	img_grad = np.zeros([rows, cols])
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

			img_grad[r-1 ,c-1] = (np.sum(temp_xmat)**2 + np.sum(temp_ymat)**2)**(0.5)

	return img_grad

'''
------------------------------------ Step 3 ------------------------------------
'''
def threshold(img_grad):
	old_t = None
	new_t = None

	rows= img_grad.shape[0]
	cols = img_grad.shape[1]

	#initialize the threshold
	new_t = np.sum(img_grad) / (img_grad.shape[0] * img_grad.shape[1])

	#determine the optimal threshold
	while(1):
		ml = []
		mh = []

		for r in range (rows):
			for c in range (cols):
				if img_grad[r,c] < new_t:
					ml.append(img_grad[r,c])

				else:
					mh.append(img_grad[r,c])

		mh = np.array(list(mh))
		ml = np.array(list(ml))

		avg_ml = np.sum(ml) / ml.shape[0]
		avg_mh = np.sum(mh) / mh.shape[0]

		old_t = new_t
		new_t = (avg_ml + avg_mh) /2

		if (new_t - old_t) <= 0:
			break

	thres_img = np.zeros([rows, cols])

	for r in range (rows):
		for c in range (cols):
			if img_grad[r,c] >= new_t:
				thres_img[r,c] = 255

	return thres_img

# ASSIGNMENT 1 PART 2 QUESTION 5 and 6

'''
return a image with each object in the image will have their own labelling
'''
def connectComponent(thres_img):
	rows= thres_img.shape[0]
	cols = thres_img.shape[1]

	queue = []
	label = 1

	pad_img = np.pad(thres_img, pad_width=1, mode='constant', constant_values=0)
	label_img = np.zeros([rows+2, cols+2])

	for r in range(1, rows+1):
		for c in range(1, cols+1):

			#if the pixel is foreground
			if pad_img[r,c] > 0:

				#check if the pixel is labelled
				if label_img[r,c] == 0: #not labelled
					label_img[r,c] = label

					queue.append((r,c))

					#check if the queue is not empty 
					if len(queue) > 0:
						num_element = len(queue)

						while (len(queue) > 0):
							row_index, col_index = queue.pop(0)

							#check whether the neighbouring pixel is foreground and is not labelled

							#upper left pixel
							if (pad_img[row_index-1, col_index-1] > 0) and (label_img[row_index-1, col_index-1] == 0):
								label_img[row_index-1, col_index-1] = label
								queue.append((row_index-1, col_index-1))

							#upper pixel
							if (pad_img[row_index-1, col_index] > 0) and (label_img[row_index-1, col_index] == 0):
								label_img[row_index-1, col_index] = label
								queue.append((row_index-1, col_index))


							#upper right pixel
							if (pad_img[row_index-1, col_index+1] > 0) and (label_img[row_index-1, col_index+1] == 0):
								label_img[row_index-1, col_index+1] = label
								queue.append((row_index-1, col_index+1))

							#left pixel
							if (pad_img[row_index, col_index-1] > 0) and (label_img[row_index, col_index-1] == 0):
								label_img[row_index, col_index-1] = label
								queue.append((row_index, col_index-1))

							#right pixel
							if (pad_img[row_index, col_index+1] > 0) and (label_img[row_index, col_index+1] == 0):
								label_img[row_index, col_index+1] = label
								queue.append((row_index, col_index+1))

							#lower left pixel
							if (pad_img[row_index+1, col_index-1] > 0) and (label_img[row_index+1, col_index-1] == 0):
								label_img[row_index+1, col_index-1] = label
								queue.append((row_index+1, col_index-1))

							#lower pixel
							if (pad_img[row_index+1, col_index] > 0) and (label_img[row_index+1, col_index] == 0):
								label_img[row_index+1, col_index] = label
								queue.append((row_index+1, col_index))

							#lower right pixel
							if (pad_img[row_index+1, col_index+1] > 0) and (label_img[row_index+1, col_index+1] == 0):
								label_img[row_index+1, col_index+1] = label
								queue.append((row_index+1, col_index+1))

					label += 1

	return label_img, label

original = cv2.imread(path, 0)

c = getGaussKernel(3, 3, 3)
gauss_img = applyGauss(c, path)
grad_img = gradMagnitude(gauss_img)
thres_img = threshold(grad_img)

#to show output for question 4, uncomment the code below

'''
plt.subplot(2,2,1)
plt.imshow(original, cmap='gray')
plt.title("input image (grayscale)")

plt.subplot(2,2,2)
plt.imshow(gauss_img, cmap='gray')
plt.title("image with gauss applied")

plt.subplot(2,2,3)
plt.imshow(grad_img, cmap='gray')
plt.title("gradient magnitude")

plt.subplot(2,2,4)
plt.imshow(thres_img, cmap='gray')
plt.title("threshold applied")
plt.show()
'''

#show output for question 6
label_img, label = connectComponent(thres_img)

print("label is ", label)
plt.imshow(label_img, cmap="gray")
plt.show()













