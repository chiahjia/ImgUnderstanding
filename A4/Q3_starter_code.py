import numpy as np
import cv2
import plotly.graph_objects as go


def get_data(folder):
	'''
	reads data in the specified image folder
	'''
	depth = cv2.imread(folder + 'depthImage.png')[:,:,0]
	rgb = cv2.imread(folder + 'rgbImage.jpg')
	extrinsics = np.loadtxt(folder + 'extrinsic.txt')
	intrinsics = np.loadtxt(folder + 'intrinsics.txt')
	return depth, rgb, extrinsics, intrinsics



def compute_point_cloud(imageNumber):
	'''
	 This function provides the coordinates of the associated 3D scene point
	 (X; Y;Z) and the associated color channel values for any pixel in the
	 depth image. You should save your output in the output_file in the
	 format of a N x 6 matrix where N is the number of 3D points with 3
	 coordinates and 3 color channel values:
	 X_1,Y_1,Z_1,R_1,G_1,B_1
	 X_2,Y_2,Z_2,R_2,G_2,B_2
	 X_3,Y_3,Z_3,R_3,G_3,B_3
	 X_4,Y_4,Z_4,R_4,G_4,B_4
	 X_5,Y_5,Z_5,R_5,G_5,B_5
	 X_6,Y_6,Z_6,R_6,G_6,B_6
	 .
	 .
	 .
	 .
	'''
	depth, rgb, extrinsics, intrinsics = get_data(imageNumber)
	# rotation matrix
	R = extrinsics[:, :3] # 3x3
	# t
	t = extrinsics[:, 3] # 3x1

# YOUR IMPLEMENTATION CAN GO HERE:
	#initilize the rotation matrix R 4x4
	new_R = np.zeros((R.shape[0] + 1, R.shape[1] + 1))
	for i in range(R.shape[0]):
		for j in range(R.shape[1]):
			new_R[i][j] = R[i][j]

	new_R[new_R.shape[0]-1 ][new_R.shape[1]-1] = 1

	#initilize [I | -c] 4x4
	ic = np.identity(4)
	for i in range (t.shape[0]):
		ic[i][3] = t[i]


	xyz_c = np.zeros((4,1))

	results = []
	
	#loop through each pixel in depthImage and rgbImage
	for i in range(depth.shape[0]):
		for j in range(depth.shape[1]):
			if depth[i][j] == 0:
				continue

			'''
			initialize the projection matrix
			[1 0 0 0]
			[0 1 0 0]
			[0 0 1 0]
			'''
			proj = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])

			#trace back the coordinate from the depthImage
			w_xy= np.array([[depth[i][j]*i],[depth[i][j]*j],[depth[i][j]]])


			#inverse the camera calibration matrix K
			inv_intrinsics = np.linalg.inv(intrinsics)

			#calculate the 3D camera coordinate system
			xyz_c = np.matmul(inv_intrinsics, w_xy)
			xyz_c_homo = np.append(xyz_c, [[1]], axis=0)

			#calculate the extrinsic matrix R[I | -c] and inversed it
			R_ic = np.matmul(new_R, ic)
			inv_R_ic = np.linalg.inv(R_ic)

			#calculate the 3D world coordinate
			xyz_w_homo = np.matmul(inv_R_ic, xyz_c_homo)


			output = [xyz_w_homo[0][0], xyz_w_homo[1][0], xyz_w_homo[2][0], rgb[i][j][0], rgb[i][j][1], rgb[i][j][2]]
			results.append(output)


	results = np.array(results)
	return results


def plot_pointCloud(pc):
	'''
	plots the Nx6 point cloud pc in 3D
	assumes (1,0,0), (0,1,0), (0,0,-1) as basis
	'''
	fig = go.Figure(data=[go.Scatter3d(
		x=pc[:, 0],
		y=pc[:, 1],
		z=-pc[:, 2],
		mode='markers',
		marker=dict(
			size=2,
			color=pc[:, 3:][..., ::-1],
			opacity=0.8
		)
	)])
	fig.show()



if __name__ == '__main__':

	imageNumbers = ['1/', '2/', '3/']
	for  imageNumber in  imageNumbers:

		# Part a)
		pc = compute_point_cloud( imageNumber)
		np.savetxt( imageNumber + 'pointCloud.txt', pc)
		plot_pointCloud(pc)

