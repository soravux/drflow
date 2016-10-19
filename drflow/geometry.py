import numpy as np
import tensorflow as tf


# Inspired by http://webcache.googleusercontent.com/search?q=cache:ccD0pZjF_HkJ:geomalgorithms.com/a06-_intersect-2.html+&cd=1&hl=fr&ct=clnk&gl=us
def intersect_triangle(p0, p1, v0, v1, v2):
    """
    p0, p1: [Nx3] begin and end of N line(s)
    v0, v1, v2: [Mx3] vertices defining M face(s)
    """

    p0 = np.atleast_2d(p0).T
    p1 = np.atleast_2d(p1).T
    v0 = np.atleast_2d(v0)
    v1 = np.atleast_2d(v1)
    v2 = np.atleast_2d(v2)

    nb_lines = p0.shape[1]
    nb_planes = v0.shape[0]
    
    # Get triangle normal
    u = v1 - v0
    v = v2 - v0
    n = np.cross(u, v)
    # n /= np.sqrt(np.sum(n**2))

    # Intersection with plane
    denom = np.dot(n, p1 - p0)
    # if np.abs(denom) < 1e-6:
    #     return np.inf

    # subtracts each row of p0 from each row of v0
    t = (np.tile(v0[:,:,np.newaxis].transpose((2,1,0)), [nb_lines,1,1]) - np.tile(p0.T[:,:,np.newaxis],[1,1,nb_planes])).transpose((0,2,1))
    d = np.sum(n[np.newaxis,:,:] * t, axis=2).T / denom

    # dref = np.dot(n, (v0 - p0.T).T) / denom
    # import pdb; pdb.set_trace()
    # if not (0 < d < 1):
    #     return np.inf

    retval_dist = np.zeros((p0.shape[1], v0.shape[0])) + np.inf
    retval_pi = np.zeros((p0.shape[1], v0.shape[0], 3))

    # Could that loop be vectorized?
    for fid in range(v0.shape[0]):

        pi = p0 + d[fid,:]*(p1-p0)

        f1 = (v0[fid,:] - pi.T)
        f2 = (v1[fid,:] - pi.T)
        f3 = (v2[fid,:] - pi.T)

        # See http://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/barycentric-coordinates
        # and http://answers.unity3d.com/questions/383804/calculate-uv-coordinates-of-3d-point-on-plane-of-m.html
        bary = np.empty((d.shape[1], 3))
        va1 = np.cross(f2, f3)
        va2 = np.cross(f3, f1)
        va3 = np.cross(f1, f2)
        bary[:,0] = np.sqrt(np.sum(va1**2, axis=va1.ndim - 1))
        bary[:,1] = np.sqrt(np.sum(va2**2, axis=va1.ndim - 1))
        bary[:,2] = np.sqrt(np.sum(va3**2, axis=va1.ndim - 1))
        bary /= np.sqrt(np.sum(n**2))
        bary *= np.array([np.sign(va1.dot(np.squeeze(n[fid,:]))), np.sign(va2.dot(np.squeeze(n[fid,:]))), np.sign(va3.dot(np.squeeze(n[fid,:])))]).T

        # Note to self: https://en.wikipedia.org/wiki/Barycentric_coordinate_system
        # bary = np.linalg.solve(t, pi)
        # print(v0, v1, v2, pi)

        valid_inter = np.all((bary > 0) & (bary < 1), axis=1)

        if valid_inter.any():
            try:
                retval_dist[valid_inter,fid] = d[fid,valid_inter]
            except:
                import pdb; pdb.set_trace()
            retval_pi[valid_inter,fid,:] = pi[:,valid_inter].T

    return np.squeeze(retval_dist), np.squeeze(retval_pi)


def get_normal_tf(v0, v1, v2, normalize=False):
    """
    Get the normal of the surface defined by the 3 vertices of the triangle (v0, v1, v2)
    """
    v0 = tf.reshape(v0, [-1, 3])
    v1 = tf.reshape(v1, [-1, 3])
    v2 = tf.reshape(v2, [-1, 3])

    u = v1 - v0
    v = v2 - v0
    n = tf.cross(u, v)

    if normalize:
        n /= tf.sqrt(tf.reduce_sum(n**2))

    return n


def intersect_triangle_tf(p0, p1, v0, v1, v2):
    """
    p0, p1: [Nx3] begin and end of N line(s)
    v0, v1, v2: [1x3] vertices defining a face
    """

    p0 = tf.reshape(p0, [-1, 3])
    p1 = tf.reshape(p1, [-1, 3])
    v0 = tf.reshape(v0, [-1, 3])
    v1 = tf.reshape(v1, [-1, 3])
    v2 = tf.reshape(v2, [-1, 3])

    p0 = tf.transpose(p0)
    p1 = tf.transpose(p1)
    
    # Get triangle normal
    n = get_normal_tf(v0, v1, v2)

    # Intersection with plane
    denom = tf.matmul(n, p1 - p0)
    # if np.abs(denom) < 1e-6:
    #     return np.inf

    # v0 = tf.Print(v0, [tf.shape(v0), tf.shape(p0)] )

    # Supports only one face or one ray as of now
    if v0.get_shape()[0] == 1:
        d = tf.reshape(tf.matmul(n, tf.transpose(v0 - tf.transpose(p0))) / denom, [1, -1])
    else:
        # d = tf.reshape(tf.reduce_sum(tf.mul(n, (v0 - tf.transpose(p0))), reduction_indices=1) / denom, [1, -1])
        # d = tf.reshape(tf.reduce_sum(tf.mul(n, (v0 - tf.transpose(p0))), reduction_indices=1) / denom, [1, -1])
        d = tf.reshape(tf.reduce_sum(tf.mul(n, (v0 - tf.transpose(p0))), reduction_indices=1) / tf.reshape(denom, [-1]), [1, -1])

    # print(n.get_shape())
    # print(v0.get_shape())
    # print(p0.get_shape())
    # print(denom.get_shape())
    # print(d.get_shape())

    # d = tf.Print(d, [tf.shape(d)])

    # if not (0 < d < 1):
    #     return np.inf

    # Pack contortions because v0.shape[0] can be unknown
    retval_dist = tf.fill(tf.pack([p0.get_shape()[1], tf.shape(v0)[0]]), tf.constant(0, dtype=tf.float32)) + tf.constant(np.inf, dtype=tf.float32)
    retval_pi = tf.fill(tf.pack([p0.get_shape()[1], tf.shape(v0)[0], 3]), tf.constant(0, dtype=tf.float32))
    # retval_dist = tf.zeros(p0.get_shape()[1], v0.get_shape()[0]) + np.inf
    # retval_pi = tf.zeros((p0.get_shape()[1], v0.get_shape()[0], 3))
    

    pi = tf.transpose(p0 + d*(p1-p0))

    f1 = (v0 - pi)
    f2 = (v1 - pi)
    f3 = (v2 - pi)

    bary = tf.fill(tf.pack([tf.shape(d)[1], 3]), 0) #tf.zeros((d.get_shape()[1], 3))
    va1 = tf.cross(f2, f3)
    va2 = tf.cross(f3, f1)
    va3 = tf.cross(f1, f2)
    b0 = tf.sqrt(tf.reduce_sum(va1**2, reduction_indices=1))
    b1 = tf.sqrt(tf.reduce_sum(va2**2, reduction_indices=1))
    b2 = tf.sqrt(tf.reduce_sum(va3**2, reduction_indices=1))
    bary = tf.transpose(tf.reshape(tf.concat(0, [b0, b1, b2]) / tf.sqrt(tf.reduce_sum(n**2)), [3, -1]))

    # if v0.get_shape()[0] == 1:
    #     bary *= tf.concat(1, [tf.sign(tf.matmul(va1, tf.transpose(n))), tf.sign(tf.matmul(va2, tf.transpose(n))), tf.sign(tf.matmul(va3, tf.transpose(n)))])
        # # bary = tf.Print(bary, [tf.shape(tf.concat(1, [tf.sign(tf.matmul(va1, tf.transpose(n))), tf.sign(tf.matmul(va2, tf.transpose(n))), tf.sign(tf.matmul(va3, tf.transpose(n)))]))])
    # else:
    # Remove mul to speed up?
    sps = lambda x, y: tf.sign(tf.reshape(tf.reduce_sum(tf.mul(x, y), reduction_indices=1), [-1, 1]))
    # bary = tf.Print(bary, [tf.shape(bary), tf.shape(sps(va1, n))])
    bary *= tf.concat(1, [sps(va1, n), sps(va2, n), sps(va3, n)])
    # deuxieme = tf.concat(1, [tf.sign(tf.matmul(va1, tf.transpose(n))), tf.sign(tf.matmul(va2, tf.transpose(n))), tf.sign(tf.matmul(va3, tf.transpose(n)))])
    # bary = tf.Print(bary, [tf.shape(va1), tf.shape(n), tf.shape(sps(va1, n)), tf.shape(tf.sign(tf.matmul(va1, tf.transpose(n))))], summarize=6)
    # bary = tf.Print(bary, [tf.shape(tf.matmul(va1, tf.transpose(n))), tf.shape(tf.reduce_sum(tf.mul(va1, n), reduction_indices=1)),
    #     tf.reduce_sum(tf.abs(tf.sign(tf.matmul(va2, tf.transpose(n))) - tf.sign(tf.reshape(tf.reduce_sum(tf.mul(va2, n), reduction_indices=1), [-1, 1]))))], summarize=6)

    # Contorsions to get the scattering / selection / numpy indexing
    d_valid = tf.logical_and(tf.greater(d, tf.constant(0, dtype=tf.float32)), tf.less(d, tf.constant(1, dtype=tf.float32)))
    bary_valid = tf.reduce_all(tf.logical_and((tf.greater(bary, tf.constant(0, dtype=tf.float32))), tf.less(bary, tf.constant(1, dtype=tf.float32))), reduction_indices=1)
    valid_inter = tf.squeeze(tf.logical_and(d_valid, bary_valid))
    # valid_idx = tf.where(valid_inter)
    # invalid_idx = tf.where(tf.logical_not(valid_inter))

    # invalid_d_vals = tf.squeeze(tf.gather(tf.squeeze(d), invalid_idx, name="dist_gather"))
    # # I want to make a SparseTensor with multiple indices (invalid_idx) set to a single value, infinity.
    # # # But I can't do that directly, so I add a tensor of the right size, let the broadcasting do its job and then set the value.
    # invalid_d_vals = tf.SparseTensor(tf.squeeze(invalid_idx), tf.reshape(tf.cast(invalid_idx, tf.float32) + tf.constant([np.inf], dtype="float32"), [-1]), tf.cast(tf.shape(tf.squeeze(d)), tf.int64))
    # out_delta = tf.sparse_tensor_to_dense(invalid_d_vals)
    # out_dist = d + out_delta

    d = tf.squeeze(d)
    out_dist = tf.select(valid_inter, d, d + tf.constant([np.inf]))

    # out_pi = tf.zeros_like(pi)
    # valid_pi_vals = tf.reshape(tf.gather(pi, valid_idx), [-1])
    # valid_idx = tf.tile(tf.reshape(valid_idx, [-1, 1]), [1, 3])
    # idx_cols = tf.zeros_like(valid_idx) + tf.cast(tf.range(3), dtype=tf.int64)
    # indices = tf.concat(1, [tf.reshape(valid_idx, [-1, 1]), tf.reshape(idx_cols, [-1, 1])])
    # valid_pi_vals_sparse = tf.SparseTensor(indices, valid_pi_vals, tf.cast(tf.shape(pi), tf.int64))
    # out_pi = tf.sparse_tensor_to_dense(valid_pi_vals_sparse)

    valid_inter_2d = tf.reshape(valid_inter, [-1, 1])
    out_pi = tf.select(tf.concat(1, [valid_inter_2d, valid_inter_2d, valid_inter_2d]), pi, tf.zeros_like(pi))

    # print(tf.gradients(pi, v0))

    return tf.squeeze(out_dist), out_pi


# From https://gist.github.com/rossant/6046463
def intersect_plane(O, D, P, N):
    # Return the distance from O to the intersection of the ray (O, D) with the 
    # plane (P, N), or +inf if there is no intersection.
    # O and P are 3D points, D and N (normal) are normalized vectors.
    denom = np.dot(D, N)
    if np.abs(denom) < 1e-6:
        return np.inf
    d = np.dot(P - O, N) / denom
    if d < 0:
        return np.inf
    return d