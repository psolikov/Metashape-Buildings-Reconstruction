import PhotoScan as ps
import codecs, json
# import numpy as np

def deserrialize_cnt(path):
    obj_text = codecs.open(path, 'r', encoding='utf-8').read()
    b_new = json.loads(obj_text)
    # a_new = np.array(b_new)
    return b_new

if __name__ == "__main__":
    # Loading names of cameras containing school that I picked manuallly
    file = open('//psdevscns/ps_storage/solikov/Segmentation/Chunk/selected_5')
    selected = json.load(file)
    doc = ps.app.document
    chunk = doc.chunk
    cameras = chunk.cameras

    print('Loading cameras...')

    selected_cameras = []
    for i in cameras:
        if (i.label in selected):
            f = location + i.label[:len(i.label) - 4] + '.JPG'
            print(f)
            selected_cameras.append(i)

    # Loading pictures of segmented contours
    # selected_contours = []
    # location = '//psdevscns/ps_storage/solikov/Segmentation/Chunk/'
    # for i in selected_cameras:
    #     f = location + i.label[:len(i.label) - 4] + '_contour.JPG'
    #     print(f)
    #     selected_contours.append(cv2.imread(f, -1))

    print('Done. Call \'print_available_cameras()\' to get list of cameras, ' +
        '\'find_contour_3d()\' to generate vertices for new shape and '+
        '\'create_new_shape()\' to create shape and show it.')

    def print_available_cameras():
        for i in range(len(selected_cameras)):
            print('id: ' + str(i) + ', name: ' + '\'' + selected_cameras[i].label + '\'')

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

    def renderDepth(cam):
        model = chunk.model
        return model.renderDepth(cam.transform, cam.sensor.calibration)

    def export_selected_cameras_to_json():
        s_list = []
        for c in chunk.cameras:
            if (c.selected):
                s_list.append(c.label)
        path_to_selected_cameras = '//psdevscns/ps_storage/solikov/Segmentation/Chunk/export_selected_cameras_to_json.txt'
        with open(path_to_selected_cameras, 'w') as outfile:
            json.dump(s_list, outfile)