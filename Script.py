import PhotoScan as ps
import json
import numpy as np
import cv2

# path to this file local: C:\Users\p.solikov\Work\FromSegmentation\Semi-Automatic\Script.py
# C:/Users/p.solikov/Work/FromSegmentation/Semi-Automatic/Script.py

file = open('C:/Users/p.solikov/Work/FromSegmentation/Semi-Automatic/selected_5')
selected = json.load(file)
doc = ps.app.document
chunk = doc.chunk
cameras = chunk.cameras
selected_cameras = []
for i in cameras:
	if (i.label in selected):
		selected_cameras.append(i)
selected_contours = []
location = 'C:/Users/p.solikov/Work/FromSegmentation/Semi-Automatic/'
print('Loading contours')
for i in selected_cameras:
	f = location + i.label[:len(i.label) - 4] + '_contour.JPG'
	selected_contours.append(cv2.imread(f, 0))
ps_cam = selected_cameras[0]
mask = ps.Mask()
mask.setImage(ps_cam.image())
mask_image = mask.image()
mask_image_byte = mask_image.tostring()
mask_image_byte = np.asarray(list(mask_image_byte), dtype='uint8')
mask_image_byte = mask_image_byte.reshape((4000, 6000))
# self.mask = Mask(mask_image_byte)

def save_contour(name):
	im2, contours, hierarchy = cv2.findContours(selected_contours[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	cv2.imwrite(name, cv2.drawContours(im2, contours, -1, (0,255,0), 3))

def find_contour_3d(points):
    # self._get_mask()
    # contour_2d = self.mask.find_contour()
    # contour_2d[contour_2d != 0] = 1
    # contour_2d_dots = list(zip(*np.where(contour_2d == 1)))
    im2, contours, hierarchy = cv2.findContours(selected_contours[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    point_cloud = chunk.dense_cloud
    contour_2d = contours
    # for (x, y) in contour_2d_dots:
    #     pixel = contour_2d[x][y]
    #     pixel_vector = ps.Vector([x, y])
    #     destination_camera = ps_cam.sensor.calibration.unproject(pixel_vector)  # 3d point on ray (camera space)
    #     destination_chunk = ps_cam.transform.mulp(destination_camera)  # 3d point on ray (chunk space)
    #     target_point = point_cloud.pickPoint(ps_cam.center, destination_chunk)  # 3d point (chunk space)
    #     if target_point is not None:
    #         world = chunk.transform.matrix.mulp(target_point)  # get geocentric coords
    #         projected = chunk.shapes.crs.project(world) #get geographic coords
    #         # projected = np.asarray(projected)
    #         # points.append(projected)    	
    #         shp = chunk.shapes.addShape()
    #         shp.vertices =  list(projected)

    for pixel_list in contour_2d:
        for pixel in pixel_list:
            y, x = pixel[0]
            pixel_vector = ps.Vector([x, y])
            destination_camera = ps_cam.sensor.calibration.unproject(pixel_vector)  # 3d point on ray (camera space)
            destination_chunk = ps_cam.transform.mulp(destination_camera)  # 3d point on ray (chunk space)
            target_point = point_cloud.pickPoint(ps_cam.center, destination_chunk)  # 3d point (chunk space)
            if target_point is not None:
                world = chunk.transform.matrix.mulp(target_point)  # get geocentric coords
                projected = chunk.shapes.crs.project(world) #get geographic coords
                projected = np.asarray(projected)
                points.append(projected)
    return points

    # for i in range(len(contour_2d)):
    #     for j in range(len(contour_2d[i])):
    #         pixel = contour_2d[i][j]
    #         if (pixel == 0):
    #             print(pixel)
    #             continue
    #         print(pixel)
    #         pixel_vector = ps.Vector([i, j])
    #         destination_camera = ps_cam.sensor.calibration.unproject(pixel_vector)  # 3d point on ray (camera space)
    #         destination_chunk = ps_cam.transform.mulp(destination_camera)  # 3d point on ray (chunk space)
    #         target_point = point_cloud.pickPoint(ps_cam.center, destination_chunk)  # 3d point (chunk space)
    #         if target_point is not None:
    #             world = chunk.transform.matrix.mulp(target_point)  # get geocentric coords
    #             projected = chunk.crs.project(world) #get geographic coords
    #             projected = np.asarray(projected)
    #             points.append(projected)
    # return points