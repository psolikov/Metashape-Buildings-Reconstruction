# Select all cameras that needed to update

def rebuild_depth_maps(model):
    import Metashape as ms
    chunk = ms.app.document.chunk
    cameras = chunk.cameras
    new_selected_cameras = []
    for c in chunk.depth_maps.keys():
      # if (c.selected):
        new_selected_cameras.append(c)
    # pass model
    for c in new_selected_cameras:
      dm = chunk.depth_maps[c]
      dm.setImage(model.renderDepth(c.transform, c.calibration))

    print('Done')