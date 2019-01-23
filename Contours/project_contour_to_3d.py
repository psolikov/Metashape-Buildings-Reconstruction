import PhotoScan as ps
import codecs, json

# Home folder: '\\psdevscns\ps_storage\solikov\kazan_part2\Data'
def get_default_folder():
    return '//psdevscns/ps_storage/solikov/kazan_part2/Data/'

def deserrialize_cnt(path):
    obj_text = codecs.open(path, 'r', encoding='utf-8').read()
    b_new = json.loads(obj_text)
    # a_new = np.array(b_new)
    return b_new

if __name__ == "__main__":
    # Loading names of cameras containing school that I picked manuallly
    home = get_default_folder()
    file = open(home + 'selected.txt')
    selected = json.load(file)
    doc = ps.app.document
    chunk = doc.chunk
    cameras = chunk.cameras

    print('Loading cameras...')

    selected_cameras = []
    location = home
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

    def get_contour_3d_2(camera, contour):
        doc = Metashape.app.document
        if not len(doc.chunks):
            raise Exception("No chunks!")

        print("Script started...")
        chunk = doc.chunk

        if not chunk.shapes:
            chunk.shapes = Metashape.Shapes()
            chunk.shapes.crs = chunk.crs
        T = chunk.transform.matrix
        footprints = chunk.shapes.addGroup()
        footprints.label = "Contours"
        footprints.color = (30, 239, 30)

        if chunk.model:
            surface = chunk.model
        elif chunk.dense_cloud:
            surface = chunk.dense_cloud
        else:
            surface = chunk.point_cloud

        if not camera.transform:
            print('No camera\'s transform!')
            return

        sensor = camera.sensor
        points = list()
        for i in contour:
            points.append(surface.pickPoint(camera.center, camera.transform.mulp(sensor.calibration.unproject(Metashape.Vector(i[0])))))
            if not points[-1]:
                points[-1] = chunk.point_cloud.pickPoint(camera.center, camera.transform.mulp(sensor.calibration.unproject(Metashape.Vector(i[0]))))
            if not points[-1]:
                break
            points[-1] = chunk.crs.project(T.mulp(points[-1]))

        # points = set_height_to_min(points)

        # for i in range(len(points)):
            # points[i] = chunk.crs.project(T.mulp(points[i]))

        if not all(points):
            print("Skipping camera " + camera.label)
            return

        if len(points) >= 4:
            points = set_height_to_avg(chunk, points, points[0])
            shape = chunk.shapes.addShape()
            shape.label = camera.label
            shape.attributes["Photo"] = camera.label
            shape.type = Metashape.Shape.Type.Polygon
            shape.group = footprints
            shape.vertices = points
            shape.has_z = True

        Metashape.app.update()
        print("Script finished!")
        return points


    # Create new shape and place it on point cloud
    def create_new_shape(label, vertices):
        shapes = chunk.shapes
        sh = shapes.addShape()
        sh.label = label
        sh.vertices = vertices
        return sh

    # def metric(x, y, z):
        # return 

    def set_height_to_min(contour_3d):
        min_norm = contour_3d[0].norm2()
        cnt_cp = contour_3d.copy()
        for i in contour_3d:
            if (i.norm() < min_norm):
                min_norm = i.norm2()
        for i in cnt_cp:
            v = [i.x, i.y, i.z]
            v[0] /= i.norm2()
            v[0] *= min_norm
            v[1] /= i.norm2()
            v[1] *= min_norm
            v[2] /= i.norm2()
            v[2] *= min_norm 
            i.x = v[0]
            i.y = v[1]
            i.z = v[2]
        return cnt_cp

    def set_height_to_avg(chunk, contour_3d, center):
        mat = chunk.crs.localframe(center)
        mat_inv = mat.inv()
        l = []
        cnt_cp = contour_3d.copy()
        for i in contour_3d:
            l.append(mat.mulp(i).z)
        avg = sum(l)/len(l)
        for i in cnt_cp:
            new_i = mat.mulp(i)
            new_i.z = avg
            new_i_proj = mat_inv.mulp(new_i)
            i.x = new_i_proj.x
            i.y = new_i_proj.y
            i.z = new_i_proj.z
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

    def process_cnt(id, cnt_in, name='probe'):
        out_cnt = get_contour_3d(id, cnt_in)
        out_cnt = set_height_to_min(out_cnt)
        sh = create_new_shape(name, out_cnt)
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