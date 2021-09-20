from skimage import data
from skimage import io
from skimage import color
from matplotlib import pyplot as plt
import cv2
import numpy as np
from numpy import linalg as la
import math

path = "/Users/chiah/Documents/csc420/A2/ex1.jpg"

row_target = 966
col_target = 1400

#calculate the gradient magnitude
def gradMagnitude(input_img):
	rows = input_img.shape[0]
	cols = input_img.shape[1]

	sobel_x_filter = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
	sobel_y_filter  = [[-1, -2, -1], [0,0,0], [1,2,1]]

	img_grad = np.zeros([rows, cols])
	pad_img = np.pad(input_img, pad_width=1, mode='constant', constant_values=0)

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

#function to find the optimal path
def find_min_path(img_grad):
	rows = img_grad.shape[0]
	cols = img_grad.shape[1]

	#create a table to keep track of the energy level and initilize them with infinity value 
	energy_table = np.full(img_grad.shape, np.inf) 
	#initialize the first row of the table with the calculated gradient value from the previous function
	energy_table[0] = img_grad[0]

	path_table = np.empty_like(img_grad, dtype=object)

	top_left_e = None
	top_center_e = None
	top_right_e = None

	for r in range(1, rows):
		for c in range(cols):
			if c-1 >= 0:
				top_left_e = energy_table[r -1 , c -1]
			else:
				top_left_e = np.inf

			if c + 1 < cols:
				top_right_e = energy_table[r-1, c + 1]
			else:
				top_right_e = np.inf

			top_center_e = energy_table[r-1, c]

			energy_table[r,c] = img_grad[r,c] + min(top_left_e, top_center_e, top_right_e)

			if top_left_e == min(top_left_e, top_center_e, top_right_e):
				path_table[r, c] = (r-1, c-1)

			elif top_center_e == min(top_left_e, top_center_e, top_right_e):
				path_table[r,c] = (r-1, c)

			else:
				path_table[r,c] = (r-1, c+1)

	min_c = np.inf
	min_c_index = 0
	for c in range(cols):
		if energy_table[rows-1, c] < min_c:
			min_c = energy_table[rows-1, c]
			min_c_index = c

	return path_table, (rows-1, min_c_index)

#wrapper function to setup and carve the image
def process_img(image, gray_img):

	img_grad = gradMagnitude(gray_img)
	path_table, seam_index = find_min_path(img_grad)

	carve_img = np.zeros_like(image[:,:-1,:])
	
	while True:
		i, j = seam_index[0], seam_index[1]

		carve_img[i,:,0] = np.delete(image[i,:,0], j)
		carve_img[i,:,1] = np.delete(image[i,:,1], j)
		carve_img[i,:,2] = np.delete(image[i,:,2], j)

		if i != 0: 
			seam_index = path_table[i,j]
		else:       
			break

	return carve_img


ori_img = cv2.imread(path)
ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
gray_img = cv2.imread(path, 0)

#used to carve image (columns)
if ori_img.shape[1] != col_target:
	carve_img = process_img(ori_img, gray_img)

	while carve_img.shape[0] != col_target:
		gray_img = cv2.cvtColor(carve_img, cv2.COLOR_RGB2GRAY)
		carve_img = process_img(carve_img, gray_img)


#used to carve image (rows)
if ori_img.shape[0] != row_target:
	ori_img = np.transpose(ori_img)
	gray_img = np.transpose(gray_img)

	carve_img = process_img(ori_img, gray_img)

	while carve_img.shape[1] != col_target:
		gray_img = cv2.cvtColor(carve_img, cv2.COLOR_RGB2GRAY)
		carve_img = process_img(carve_img, gray_img)

plt.imshow(carve_img)
plt.title("carve_img")
plot.show()



