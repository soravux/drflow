# Usual imports
import time
import numpy as np
import tensorflow as tf

from loaders import OBJ
from geometry import intersect_triangle, intersect_triangle_tf, get_normal_tf
from rotlib import rotate

# Inspirations:
# OpenDR: https://github.com/mattloper/opendr/blob/master/renderer.py
# Reversible-raytracer: https://github.com/lebek/reversible-raytracer
#

###
# Cameras
###

def orthographicCameraIntrinsics(fx=1, fy=1, cx=0, cy=0):
    return np.array([[fx, 0, cx, 0],
                     [0, fy, cy, 0],
                     [0, 0, 0, 1]])


def projectiveCameraIntrinsics(f):
    return np.array([[fx, 0, cx, 0],
                     [0, fy, cy, 0],
                     [0, 0, 1, 0]])


def extrinsics():
    import rotlib
    out = (rotlib.EV2DCM([1, 0, 0], np.pi/24) \
        +  np.array([[0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]]))
    return out

    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


def loadDemoGeometry():
    geometry = OBJ('bunny_lo.obj', swapyz=True)
    vertices = np.array(geometry.vertices) / 2
    vertices = rotate(vertices, 'EV', [1, 0, 0, np.pi/2], point=np.mean(vertices, axis=0))
    vertices = rotate(vertices, 'EV', [0, 0, 1, -np.pi/2], point=np.mean(vertices, axis=0))
    vertices += np.array([[0.1, 0.05, 5]])
    faces = np.array(list(zip(*geometry.faces))[0]) - 1 # 0-indexed in Python
    albedo = np.ones((faces.shape[0], 3))

    return vertices.astype(np.float32), faces.astype(np.int32), albedo.astype(np.float32)


def loadDemoLighting():
    light_pos = np.array([[10, 0, 0],
                          [0, 4, 0],
                          [-2, -2, 0]], dtype=np.float32)
    light_intensity = np.array([[0.7, 0, 0],
                                [0, 0.7, 0],
                                [0, 0, 0.7]], dtype=np.float32)

    return light_pos, light_intensity


def buildRefRenderingPipeline():
    """
    This is the reference raytracer, made in numpy.
    """

    # Geometry
    vertices, faces, albedo = loadDemoGeometry()

    # Lighting
    light_pos, light_intensity = loadDemoLighting()

    # Camera
    cameraIntrinsics = orthographicCameraIntrinsics()
    cameraExtrinsics = extrinsics()
    cameraProjection = cameraIntrinsics.dot(cameraExtrinsics)
    cameraUnprojection = np.linalg.pinv(cameraProjection)[:3,:3]

    # Film (Pixel coordinates)
    far = 10
    pixels = np.vstack([m.ravel() for m in np.mgrid[-0.5:0.5:80j,-0.5:0.5:140j]] + [far*np.ones((80, 140)).ravel()]).T
    nb_pixels = pixels.shape[0]

    # Slower -- caches / size of intermediate tables?
    # inter_d, inter_pi = intersect_triangle(np.hstack((pixels[:,:2], np.zeros((nb_pixels, 1)))), pixels, vertices[faces[:,0],:],
    #                                                                                                     vertices[faces[:,1],:],
    #                                                                                                     vertices[faces[:,2],:])

    # dist = np.min(inter_d, axis=1)
    # closest = np.argmin(inter_d, axis=1)
    # pi = inter_pi[np.arange(inter_pi.shape[0]),closest,:]

    # closest[~np.isfinite(dist)] = -1

    dist = np.zeros((nb_pixels,)) + np.inf
    pi = np.zeros((nb_pixels,3))
    closest = np.zeros((nb_pixels,)) - 1

    for fidx, face in enumerate(faces):
        # TODO: precompute n, denom, ...
        inter_d, inter_pi = intersect_triangle(np.hstack((pixels[:,:2], np.zeros((nb_pixels, 1)))), pixels, vertices[face[0],:],
                                                                                                      vertices[face[1],:],
                                                                                                      vertices[face[2],:])
        closer_obj = (inter_d < dist) & (inter_d > 0)
        closest[closer_obj] = fidx
        dist[closer_obj] = inter_d[closer_obj]
        pi[closer_obj,:] = inter_pi[closer_obj,:]


    # output color
    color = np.zeros((nb_pixels, 3))

    for pidx, surf in enumerate(closest):
        if surf < 0:
            continue

        # print("Lighting {}/{}".format(pidx, closest.shape[0]))

        for lidx, light in enumerate(light_pos):
            # Get surface normal (Already done in intersect_triangle, btw)
            u = vertices[faces[surf,1],:] - vertices[faces[surf,0],:]
            v = vertices[faces[surf,2],:] - vertices[faces[surf,0],:]
            n = np.cross(u, v)
            n /= np.sqrt(np.sum(n**2))

            l = light - pi[pidx,:]
            l /= np.sqrt(np.sum(l**2))

            # check for shadows (occlusions)
            # Technically, we could do them all at once
            inter_d, _ = intersect_triangle(pi[pidx,:], light, vertices[faces[:,0],:], vertices[faces[:,1],:], vertices[faces[:,2],:])
            # print(inter_d[np.isfinite(inter_d)].min(), inter_d[np.isfinite(inter_d)].max())
            if not np.any((inter_d < 1) & (inter_d > 1e-3)):
                color[pidx,:] += albedo[surf,:] * light_intensity[lidx,:] * np.abs(n.dot(l))
            # color[pidx,:] = light_intensity[lidx,:] * np.abs(n.dot(l))

    from matplotlib import pyplot as plt
    plt.subplot(121); plt.imshow(dist.reshape(80,140))
    plt.subplot(122); plt.imshow(color.reshape(80,140,3))
    plt.show()


    import pdb; pdb.set_trace()



def buildRenderingPipeline():
    """
    This is the TensorFlow raytracer, modeled after the numpy version.
    """

    # Time to build up some graph!

    # Setting up inputs
    # Camera
    cameraIntrinsics = tf.placeholder(tf.float32, shape=(3, 4), name="CameraIntrinsics")
    cameraExtrinsics = tf.placeholder(tf.float32, shape=(4, 4), name="CameraExtrinsics")

    # Geometry
    vertices = tf.placeholder(tf.float32, shape=(None, 3), name="Vertices")
    faces = tf.placeholder(tf.int32, shape=(None, 3), name="Faces")
    albedo = tf.placeholder(tf.float32, shape=(None, 3), name="Albedo")

    # Lighting
    light_pos = tf.placeholder(tf.float32, shape=(None, 3), name="LightingPos")
    light_intensity = tf.placeholder(tf.float32, shape=(None, 3), name="LightingIntensity")

    # Build the unprojection matrix
    # Rendering pipeline
    cameraProjection = tf.matmul(cameraIntrinsics, cameraExtrinsics)
    # Unprojection is the transpose of projection
    # cameraUnprojection = tf.transpose(cameraProjection)
    cameraUnprojection = tf.concat(0, [tf.transpose(cameraProjection[:3,:3]), tf.reshape(-cameraProjection[:,3], [1, -1])])

    # width, height = 140, 80
    width, height = 180, 120

    # Film (Pixel coordinates)
    far = tf.constant(10., dtype=tf.float32)
    # meshgrid seems broken, it builds an NxN and an MxM grid instead of 2x NxM grids
    # U, V = tf.meshgrid(tf.linspace(-0.5, 0.5, 80), tf.linspace(-0.5, 0.5, 140))
    U = tf.reshape(tf.linspace(-0.5, 0.5, width), [1, -1])
    V = tf.reshape(tf.linspace(-0.5, 0.5, height), [-1, 1])
    U = tf.reshape(tf.tile(U, [height, 1]), [-1,])
    V = tf.reshape(tf.tile(V, [1, width]), [-1,])
    viewport = tf.reshape(tf.concat(0, [V, U, tf.ones([height*width,], tf.float32)]), [3, -1])
    pixels_origins = tf.transpose(viewport)

    nb_pixels = tf.shape(pixels_origins)[0]
    nb_faces = tf.shape(faces)[0]
    nb_lights = tf.shape(light_pos)[0]

    # pixels = tf.concat(1, [pixels_origins[:,:2], tf.zeros([nb_pixels, 1]) + far])

    # Ref: http://math.umn.edu/~poli0048/CameraModels.pdf

    # Apply camera (TODO: TEST THAT)
    # pixels_origins = tf.transpose(tf.matmul(tf.transpose(cameraProjection), tf.transpose(tf.concat(1, [tf.transpose(viewport)[:,:2], tf.ones([nb_pixels, 1])]))))
    pixels_origins = tf.transpose(tf.matmul(cameraProjection, tf.concat(0, [viewport[:2,:], tf.zeros([1, nb_pixels]), tf.ones([1, nb_pixels])])))
    # pixels_origins = tf.Print(pixels_origins, [pixels_origins], summarize=10)
    # pixels_origins = tf.transpose(tf.transpose(pixels_origins[:,:3]) / pixels_origins[:,3]) # Homogeneous coordinates
    # print(cameraUnprojection.get_shape())
    # pixels = tf.matmul(tf.concat(1, [pixels, tf.ones([nb_pixels, 1])]), tf.transpose(cameraUnprojection))
    # [X,Y,Z] = [X/Z, Y/Z]
    # Get the far plane

    # pixels = tf.matmul(cameraIntrinsics, tf.transpose(tf.concat(1, [tf.transpose(viewport), tf.ones([nb_pixels, 1])])))
    # pixels = tf.Print(pixels, [pixels], summarize=10)
    # pixels = pixels[:2,:] / pixels[2,:]
    # pixels = tf.Print(pixels, [pixels, tf.concat(0, [pixels, far*tf.ones([1, nb_pixels]), tf.ones([1, nb_pixels])])], summarize=10)

    # pixels = tf.transpose(tf.matmul(cameraProjection, tf.concat(0, [viewport[:2,:], 1*tf.ones([1, nb_pixels]), tf.ones([1, nb_pixels])])))

    # pixels = tf.Print(pixels, [pixels], summarize=10)
    # pixels = tf.transpose(tf.transpose(pixels[:,:3]) / pixels[:,3]) # Homogeneous coordinates
    # pixels = tf.transpose(pixels)

    pixels_origins = tf.concat(1, [pixels_origins[:,:2], tf.zeros([nb_pixels, 1])])

    pixels = pixels_origins + far*tf.constant([0, 0, 1], dtype=tf.float32)
    

    # pixels = tf.matmul(tf.concat(1, [pixels_origins, tf.ones([nb_pixels, 1])]), tf.transpose(cameraProjection))
    # pixels = tf.matmul(tf.concat(1, [tf.transpose(viewport)[:,:2], tf.ones([nb_pixels, 1])]), cameraUnprojection)
    # pixels *= far
    # pixels = tf.transpose(tf.transpose(pixels[:,:3]) / pixels[:,3]) # Homogeneous coordinates
    # pixels = pixels_origins + far*(pixels - pixels_origins)
    pixelsr = pixels

    # pixelsr = pixels

    dist = tf.zeros((nb_pixels,)) + tf.constant(np.inf, dtype=tf.float32)
    pi = tf.zeros((nb_pixels,3))
    closest = tf.zeros((nb_pixels,), dtype=tf.int32) - tf.constant(1, dtype=tf.int32)

    # Step 1)
    # Cast the rays on the scene

    def detectIntersections(dist, pi, closest_, n):
        face = tf.gather(faces, n, name="faceGather")
        v = tf.gather(vertices, face, name="VertexGather")

        inter_d, inter_pi = intersect_triangle_tf(pixels_origins, pixels, v[0,:], v[1,:], v[2,:])

        closer_obj = tf.logical_and(tf.less(inter_d, dist), tf.greater(inter_d, tf.constant(0, dtype=tf.float32)))
        # closer_obj_idx_2d = tf.where(closer_obj)
        # closer_obj_idx = tf.squeeze(closer_obj_idx_2d)

        dist = tf.select(closer_obj, inter_d, dist)
        # Update intersection distance
        # aug_d = tf.concat(1, [tf.expand_dims(dist, 1), tf.expand_dims(inter_d, 1)])
        # dist = tf.reduce_min(aug_d, reduction_indices=1)

        # Updating closest
        # Incredible contortions to workaround limitations / bugs in TensorFlow.
        # Check in geometry.py for more info
        # face_ids = tf.constant(0, dtype=tf.int32) * tf.reshape(tf.cast(closer_obj_idx, tf.int32), [-1]) + n
        face_ids = tf.constant(0, dtype=tf.int32) * tf.reshape(tf.cast(closer_obj, tf.int32), [-1]) + n
        # valid_closest_vals = tf.SparseTensor(tf.squeeze(closer_obj_idx), face_ids, dist.get_shape()) # Dist because closest is unknown size? (what?)
        # invalid_closest_vals = tf.SparseTensor(tf.squeeze(closer_obj_idx), tf.gather(closest_, closer_obj_idx), dist.get_shape())
        # closest_delta = tf.sparse_tensor_to_dense(invalid_closest_vals) - tf.sparse_tensor_to_dense(valid_closest_vals) # TODO: sparse_to_dense direct?
        # closest_ -= closest_delta

        closest_ = tf.select(closer_obj, face_ids, closest_)

        # Updating projections / points of intersection
        # inter_pi_select = tf.gather(inter_pi, closer_obj_idx)

        # indices = tf.tile(closer_obj_idx_2d, [1, 3])
        # idx_cols = tf.zeros_like(indices) + tf.cast(tf.range(3), dtype=tf.int64)
        # indices_2d = tf.concat(1, [tf.reshape(indices, [-1, 1]), tf.reshape(idx_cols, [-1, 1])])

        # valid_pi_vals = tf.SparseTensor(indices_2d, tf.reshape(inter_pi_select, [-1]), [tf.cast(nb_pixels, tf.int64), 3])

        # invalid_pi_vals = tf.SparseTensor(indices_2d, tf.reshape(tf.gather(pi, closer_obj_idx), [-1]), [tf.cast(nb_pixels, tf.int64), 3])

        # pi_delta = tf.sparse_tensor_to_dense(invalid_pi_vals) - tf.sparse_tensor_to_dense(valid_pi_vals)
        # pi -= pi_delta

        closer_obj_2d = tf.reshape(closer_obj, [-1, 1])
        pi = tf.select(tf.concat(1, [closer_obj_2d, closer_obj_2d, closer_obj_2d]), inter_pi, pi)

        # print(tf.gradients(inter_d, vertices))
        # print(tf.gradients(tf.sparse_tensor_to_dense(valid_pi_vals), vertices))

        n += tf.constant(1, dtype=tf.int32)
        return dist, pi, closest_, n


    n = tf.Variable(0, dtype=tf.int32)
    dist_out, pi_out, closest_out, _ = tf.while_loop(lambda dist, pi, closest, n: n < nb_faces,
                                                                  detectIntersections,
                                                                  [dist, pi, closest, n],
                                                                  back_prop=True, parallel_iterations=1)
    # TODO: Set distance to zero when greater than 1 ()

    # Step 2)
    # Compute lighting

    # output color image
    color = tf.zeros((nb_pixels, 3), dtype=tf.float32)

    def eachRayCollision(color, m):

        # Hack to dismiss pixels without any intersection (set light index > number of light before the loop)
        k = nb_lights*tf.cast(tf.less(tf.gather(closest_out, m), tf.constant(0, dtype=tf.int32)), tf.int32)
        color, m, _ = tf.while_loop(lambda color, m, k: k < nb_lights, applylighting, [color, m, k], back_prop=True)

        m += tf.constant(1, dtype=tf.int32)
        return color, m


    def applylighting(color_, m, k):
        light = tf.gather(light_pos, k)

        face = tf.gather(faces, tf.gather(closest_out, m))
        v = tf.gather(vertices, face)

        n = get_normal_tf(v[0,:], v[1,:], v[2,:], normalize=True)

        l = light - tf.gather(pi_out, m)
        l /= tf.sqrt(tf.reduce_sum(l**2))

        vall0 = tf.gather(vertices, faces[:,0])
        vall1 = tf.gather(vertices, faces[:,1])
        vall2 = tf.gather(vertices, faces[:,2])

        inter_d, _ = intersect_triangle_tf(tf.gather(pi_out, m), light, vall0, vall1, vall2)
        occlusion = tf.logical_and(tf.less(inter_d, tf.constant(1, dtype=tf.float32)), tf.greater(inter_d, tf.constant(1e-3, dtype=tf.float32)))
        shadow = tf.cast(tf.logical_not(tf.reduce_any(occlusion)), tf.float32)

        refl = shadow * tf.squeeze(tf.gather(light_intensity, k) * tf.abs(tf.matmul(n, tf.reshape(l, [-1, 1]))))

        # indices = tf.cast(tf.ones([3, 1], dtype=tf.int32)*m, tf.int64)
        # idx_cols = tf.expand_dims(tf.cast(tf.range(3), dtype=tf.int64), 1)
        # indices = tf.concat(1, [indices, idx_cols])
        # refl_mat = tf.SparseTensor(indices, refl, [tf.cast(nb_pixels, tf.int64), 3])
        # color_ += tf.sparse_to_dense(indices, [tf.cast(nb_pixels, tf.int64), 3], refl)

        valid_row = tf.equal(tf.range(nb_pixels), m)
        valid_row = tf.tile(tf.reshape(valid_row, [-1, 1]), [1, 3])

        refl = tf.tile(tf.reshape(refl, [1, 3]), [nb_pixels, 1])

        color_ += tf.select(valid_row, refl, tf.zeros_like(color_))

        # print(tf.gradients(tf.sparse_tensor_to_dense(refl), light_intensity))

        k += tf.constant(1, dtype=tf.int32)
        return color_, m, k


    m = tf.Variable(0, dtype=tf.int32)
    color_out, _ = tf.while_loop(lambda color, m: m < nb_pixels, eachRayCollision, [color, m], back_prop=True)

    g1 = tf.gradients(dist_out, vertices)
    g2 = tf.gradients(color_out, light_pos)
    g3 = tf.gradients(color_out, cameraExtrinsics)

    # Fire up this baby
    init = tf.initialize_all_variables()
    print("Building graph..."); ts = time.time()
    sess = tf.Session()
    print("Done in {}s".format(time.time() - ts))
    print("Initializing variables..."); ts = time.time()
    sess.run(init)
    print("Done in {}s".format(time.time() - ts))

    # Logs the graph and results
    # This is not mandatory to get the result, but allows the use of TensorBoard
    writer = tf.train.SummaryWriter("/tmp/drflow_log", sess.graph, flush_secs=10)

    # Now we have to tell Tensorflow about the actual values we want to use!

    print("Loading data..."); ts = time.time()
    # Geometry
    vertices_data, faces_data, albedo_data = loadDemoGeometry()

    # Lighting
    light_pos_data, light_intensity_data = loadDemoLighting()

    # Camera
    cameraIntrinsics_data = orthographicCameraIntrinsics()
    cameraExtrinsics_data = extrinsics()
    print("Done in {}".format(time.time() - ts))
    
    print("Executing graph..."); ts = time.time()
    # ori, pix = sess.run([pixels_origins, pixelsr], feed_dict={cameraIntrinsics: cameraIntrinsics_data,
    #                                     cameraExtrinsics: cameraExtrinsics_data,
    #                                     light_pos: light_pos_data,
    #                                     light_intensity: light_intensity_data,
    #                                     vertices: vertices_data,
    #                                     faces: faces_data,
    #                                     albedo: albedo_data,
    #                                     })
    # import pdb; pdb.set_trace()
    dist_data, image, closest_data, g1_data, g2_data, g3_data = sess.run([dist_out, color_out, closest_out, g1, g2, g3], feed_dict={cameraIntrinsics: cameraIntrinsics_data,
                                        cameraExtrinsics: cameraExtrinsics_data,
                                        light_pos: light_pos_data,
                                        light_intensity: light_intensity_data,
                                        vertices: vertices_data,
                                        faces: faces_data,
                                        albedo: albedo_data,
                                        })
    print("Done in {}".format(time.time() - ts))

    # grad_data = sess.run(grad, feed_dict={cameraIntrinsics: cameraIntrinsics_data,
    #                                     cameraExtrinsics: cameraExtrinsics_data,
    #                                     light_pos: light_pos_data,
    #                                     light_intensity: light_intensity_data,
    #                                     vertices: vertices_data,
    #                                     faces: faces_data,
    #                                     albedo: albedo_data,
    #                                     })

    # grad_data = np.array([0])


    grad_vert = np.zeros(g1_data[0].dense_shape)
    grad_vert[g1_data[0].indices,:] = g1_data[0].values

    g2_vert = np.zeros([g2_data[0].indices.shape[0], 3])
    g2_vert[g2_data[0].indices,:] = g2_data[0].values

    g3_data = g3_data[0]
    # DenseShape 3x3????

    dist_data[dist_data > 10] = 0

    # g3_out = sess.run(g3, feed_dict={cameraIntrinsics: cameraIntrinsics_data,cameraExtrinsics: cameraExtrinsics_data,light_pos: light_pos_data,light_intensity: light_intensity_data,vertices: vertices_data,faces: faces_data,albedo: albedo_data,})

    from matplotlib import pyplot as plt
    plt.subplot(231); plt.imshow(dist_data.reshape(height,width)); plt.clim([np.nanmin(dist_data[dist_data > 0]), np.nanmax(dist_data)]); plt.title("Depth"); plt.colorbar(shrink=0.7)
    plt.subplot(232); plt.imshow(closest_data.reshape(height,width), interpolation='nearest'); plt.colorbar(shrink=0.7); plt.title("Surface index")
    plt.subplot(233); plt.imshow(image.reshape(height,width,3) / image.max()); plt.title("Render")
    plt.subplot(234); plt.imshow(grad_vert[4:24,:], interpolation='nearest'); plt.colorbar(); plt.title("d(depth)/d(vertices), 20 faces"); plt.xlabel("Components"); plt.ylabel("Face index")
    plt.subplot(235); plt.imshow(g2_vert[2000:2020,:], interpolation='nearest'); plt.colorbar(); plt.title("d(image)/d(light pos), 20 pixels"); plt.xlabel("Components"); plt.ylabel("Pixel index")
    plt.subplot(236); plt.imshow(g3_data, interpolation='nearest'); plt.colorbar(); plt.title("d(image)/d(CamExtrinsics)")
    plt.show()

    import pdb; pdb.set_trace()

    return


if __name__ == '__main__':
    buildRenderingPipeline()
    # buildRefRenderingPipeline()