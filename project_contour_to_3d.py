import PhotoScan as ps
import json
import numpy as np
import cv2
from skimage.transform import AffineTransform, warp
from skimage.color import label2rgb

# Run this on //psdevscns/ps_storage/solikov/kazan_part/P084_1710_Kazan_III-IV_E-8

# path to local json: C:/Users/p.solikov/Work/FromSegmentation/Semi-Automatic/selected_5
# path to local photos folder: C:/Users/p.solikov/Work/FromSegmentation/Semi-Automatic/

# Function to get contour of a building
# It is required to have 3 files in provided location: image with the building, image with background markers, image with foreground markers
# format: 'name.JPG', 'name_bg.JPG', 'name_fg.JPG'
# example of args: location = '//psdevscns/ps_storage/solikov/Segmentation/Chunk/', filename = '2018_08_15_Naklon-Left_g201b20265_f004_0629'
def get_contour(location, filename):
    JPG = '.JPG'
    img = cv2.imread(location + filename + JPG)
    markers = cv2.imread(location + filename + '_fg' + JPG, 0)
    markers_bg = cv2.imread(location + filename + '_bg' + JPG, 0)
    labels = np.zeros_like(markers, dtype=np.int32)
    labels[markers == 255] = 1
    labels[markers_bg == 255] = 2
    out = cv2.watershed(img, labels)
    out_cp = out.copy()
    out_cp = out_cp.astype('uint8')
    out_cp[out_cp == 2] = 255
    out_cp[out_cp == -1] = 255
    out_cp[out_cp == 1] = 0
    # contour_image = np.zeros_like(out)
    # contour_image = contour_image.reshape((contour_image.shape[0], contour_image.shape[1] , 1))
    # contour_image[out == -1] = 255
    # contour_image = contour_image.astype('uint8')
    im2, contours, hierarchy = cv2.findContours(out_cp, cv2.RETR_LIST , cv2.CHAIN_APPROX_NONE)
    cnt = None
    if (cv2.contourArea(contours[0]) < cv2.contourArea(contours[1])):
        cnt = contours[0]
    else:
        cnt = contours[1]
    return cnt

def approximate_contour(cnt):
    epsilon = 0.0002*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    return approx

if __name__ == "__main__":
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

    # Export 2d contour from segmented image with a given contour from get_contour
    def get_contour_3d(camera_index, contour):
        p = []
        ps_cam = selected_cameras[camera_index]
        point_cloud = chunk.dense_cloud
        for pixel_list in contour:
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

    def set_height_to_min(contour_3d):
        min = contour_3d[0].z
        cnt_cp = contour_3d.copy()
        for i in contour_3d:
            if (i.z < min):
                min = i.z
        for i in cnt_cp:
            i.z = min
        return cnt_cp        

    def process_files(id):
        location = '//psdevscns/ps_storage/solikov/Segmentation/Chunk/'
        filename = selected_cameras[id].label[:len(selected_cameras[id].label) - 4]
        cnt = get_contour(location, filename)
        cnt_approx = approximate_contour(cnt)
        print('Len approx: ' + str(len(cnt_approx)))
        cnt_3d = get_contour_3d(id, cnt_approx)
        cnt_3d = set_height_to_min(cnt_3d)
        sh = create_new_shape('probe', cnt_3d)
        return sh

    def delete_cnt():
        shapes = chunk.shapes
        p = shapes.shapes[1]
        shapes.remove(p)
