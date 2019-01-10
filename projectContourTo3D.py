import PhotoScan as ps
import json
import numpy as np
import cv2

# Run this on //psdevscns/ps_storage/solikov/kazan_part/P084_1710_Kazan_III-IV_E-8

# path to local json: C:/Users/p.solikov/Work/FromSegmentation/Semi-Automatic/selected_5
# path to local photos folder: C:/Users/p.solikov/Work/FromSegmentation/Semi-Automatic/

# Loading names of cameras containing school that I picked manuallly
file = open('//psdevscns/ps_storage/solikov/Segmentation/Chunk/selected_5')
selected = json.load(file)
doc = ps.app.document
chunk = doc.chunk
cameras = chunk.cameras

selected_cameras = []
for i in cameras:
    if (i.label in selected):
        selected_cameras.append(i)

print('Loading contours...')

# Loading pictures of segmented contours
selected_contours = []
location = '//psdevscns/ps_storage/solikov/Segmentation/Chunk/'
for i in selected_cameras:
    f = location + i.label[:len(i.label) - 4] + '_contour.JPG'
    print(f)
    selected_contours.append(cv2.imread(f, -1))

print('Done. Call \'print_available_cameras()\' to get list of cameras, ' +
    '\'find_contour_3d()\' to generate vertices for new shape and '+
    '\'create_new_shape()\' to create shape and show it.')

def print_available_cameras():
    for i in range(len(selected_cameras)):
        print('id: ' + str(i) + ', name: ' + '\'' + selected_cameras[i].label + '\'')

# Export 2d contour from segmented image
def find_contour_3d(camera_index):
    p = []
    ps_cam = selected_cameras[camera_index]
    im2, contours, hierarchy = cv2.findContours(selected_contours[camera_index], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_2d = contours[0]
    point_cloud = chunk.dense_cloud
    for pixel_list in contour_2d:
        for pixel in pixel_list:
            pixel_vector = ps.Vector([pixel[0], pixel[1]])
            destination_camera = ps_cam.sensor.calibration.unproject(pixel_vector)  # 3d point on ray (camera space)
            destination_chunk = ps_cam.transform.mulp(destination_camera)  # 3d point on ray (chunk space)
            target_point = point_cloud.pickPoint(ps_cam.center, destination_chunk)  # 3d point (chunk space)
            if target_point is not None:
                world = chunk.transform.matrix.mulp(target_point)  # get geocentric coords
                projected = chunk.shapes.crs.project(world) #get geographic coords
                p.append(projected)
    return p

# Create new shape and place it on point cloud
def create_new_shape(label, vertices):
    shapes = chunk.shapes
    sh = shapes.addShape()
    sh.label = label
    sh.vertices = vertices
    return sh