# Please place imports here.
# BEGIN IMPORTS
import numpy as np
import cv2
# END IMPORTS


def compute_photometric_stereo_impl(lights, images):
    """
    Given a set of images taken from the same viewpoint and a corresponding set
    of directions for light sources, this function computes the albedo and
    normal map of a Lambertian scene.

    If the computed albedo for a pixel has an L2 norm less than 1e-7, then set
    the albedo to black and set the normal to the 0 vector.

    Normals should be unit vectors.

    Input:
        lights -- 3 x N array.  Each column is a unit vector representing a 
                  light source and its orientation in all three axes.
        images -- list of N images.  Each image is of the same scene from the
                  same viewpoint, but under the lighting condition specified in
                  lights.
    Output:
        albedo -- float32 height x width x channels image with dimensions
                  matching the input images.
        normals -- float32 height x width x 3 image with dimensions matching
                   the input images.
    """
    N = len(images)
    h, w, c = images[0].shape
    I = np.reshape(images, (N, (h * w * c)))
    L = lights
    G = np.linalg.inv((L@L.T))@L@I
    p = np.linalg.norm(G, axis=0)
    rho = np.reshape(p, (h, w, c))  # to 2D --> 3D

    G_reshape = np.reshape(G, (3, h, w, c)) # 3*h*w*c
    G_prime = np.sum(G_reshape, axis = 3)
    normals = G_prime / np.maximum(1e-7, np.linalg.norm(G_prime, axis=0))
    normals = np.moveaxis(normals, 0, 2)

    cond = np.linalg.norm(rho, axis=2) < 1e-7
    rho[cond, :] = 0.0
    normals[cond, :] = 0.0

    return rho, normals

def pyrdown_impl(image):
    """
    Prefilters an image with a gaussian kernel and then downsamples the result
    by a factor of 2.

    The following 1D convolution kernel should be used in both the x and y
    directions.
    K = 1/16 [ 1 4 6 4 1 ]

    Functions such as cv2.GaussianBlur and scipy.ndimage.gaussian_filter are
    prohibited.  You must implement the separable kernel.  However, you may
    use functions such as cv2.filter2D or scipy.ndimage.correlate to do the actual
    correlation / convolution. Note that for images with one channel, cv2.filter2D
    will discard the channel dimension so add it back in.

    Filtering should mirror the input image across the border.
    For scipy this is mode = mirror.
    For cv2 this is mode = BORDER_REFLECT_101.

    Downsampling should take the even-numbered coordinates with coordinates
    starting at 0.

    Input:
        image -- height x width x channels image of type float32.
    Output:
        down -- ceil(height/2) x ceil(width/2) x channels image of type
                float32.
    """
    c = image.shape[2]
    filter = 1/16 * np.array([1, 4, 6, 4, 1 ])
    image = cv2.filter2D(src = image, ddepth = -1, kernel = filter[:, np.newaxis], borderType = cv2.BORDER_REFLECT_101)
    image = cv2.filter2D(src = image, ddepth = -1, kernel = filter[:, np.newaxis].T, borderType = cv2.BORDER_REFLECT_101)
    down = image[::2, ::2]
    if (c == 1):
        down = np.reshape(down, (down.shape[0], down.shape[1], 1))
    return down


def pyrup_impl(image):
    """
    Upsamples an image by a factor of 2 and then uses a gaussian kernel as a
    reconstruction filter.

    The following 1D convolution kernel should be used in both the x and y
    directions.
    K = 1/8 [ 1 4 6 4 1 ]
    Note: 1/8 is not a mistake.  The additional factor of 4 (applying this 1D
    kernel twice) scales the solution according to the 2x2 upsampling factor.

    Filtering should mirror the input image across the border.
    For scipy this is mode = mirror.
    For cv2 this is mode = BORDER_REFLECT_101.

    Upsampling should produce samples at even-numbered coordinates with
    coordinates starting at 0.

    Input:
        image -- height x width x channels image of type float32.
    Output:
        up -- 2*height x 2*width x channels image of type float32.
    """
    filter = 1/8 * np.array([ 1, 4, 6, 4, 1 ])
    h,w,c = image.shape
    image_up = np.zeros((2*h, 2*w, c))
    image_up[::2, ::2] = image[:, :]
    image_up = cv2.filter2D(src = image_up, ddepth = -1, kernel = filter[:, np.newaxis], borderType = cv2.BORDER_REFLECT_101)
    image_up = cv2.filter2D(src = image_up, ddepth = -1, kernel = filter[:, np.newaxis].T, borderType = cv2.BORDER_REFLECT_101)
    if (c == 1):
        image_up = np.reshape(image_up, (image_up.shape[0], image_up.shape[1], 1))
    return image_up

def project_impl(K, Rt, points):
    """
    Project 3D points into a calibrated camera.

    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- height x width x 3 array of 3D points
    Output:
        projections -- height x width x 2 array of 2D projections
    """
    P = K @ Rt
    h, w, c = points.shape
    projections = np.zeros((h, w, 2))

    temp4 = np.ones((h, w, c+1))
    temp4[:, :, 0:3] = points
    #temp3 = np.ones((h, w, 3))
    temp3 = np.tensordot(temp4, P.T, axes=1)
    # for i in range(h):
    #     for j in range(w):
    #         x_img = P @ temp4[i, j, :]
    #         temp3[i, j, :] = x_img
    # projections[:, :, 0] = temp3[:, :, 0] / temp3[:, :, 2]
    # projections[:, :, 1] = temp3[:, :, 1] / temp3[:, :, 2]
    projections = temp3 / temp3[:, :, 2:]
    return projections[:,:,:2]

def unproject_corners_impl(K, width, height, depth, Rt):
    """
    Undo camera projection given a calibrated camera and the depth for each
    corner of an image.

    The output points array is a 2x2x3 array arranged for these image
    coordinates in this order:

     (0, 0)      |  (width, 0)
    -------------+------------------
     (0, height) |  (width, height)

    Each of these contains the 3 vector for the corner's corresponding
    point in 3D.

    Tutorial:
      Say you would like to unproject the pixel at coordinate (x, y)
      onto a plane at depth z with camera intrinsics K and camera
      extrinsics Rt.

      (1) Convert the coordinates from homogeneous image space pixel
          coordinates (2D) to a local camera direction (3D):
          (x', y', 1) = K^-1 * (x, y, 1)
      (2) This vector can also be interpreted as a point with depth 1 from
          the camera center.  Multiply it by z to get the point at depth z
          from the camera center.
          (z * x', z * y', z) = z * (x', y', 1)
      (3) Use the inverse of the extrinsics matrix, Rt, to move this point
          from the local camera coordinate system to a world space
          coordinate.
          Note:
            | R t |^-1 = | R^T -R^T t |
            | 0 1 |      | 0      1   |

          p = R^T * (z * x', z * y', z) - R^T t

    Input:
        K -- camera intrinsics calibration matrix
        width -- camera width
        height -- camera height
        depth -- depth of plane with respect to camera
        Rt -- 3 x 4 camera extrinsics calibration matrix
    Output:
        points -- 2 x 2 x 3 array of 3D points
    """
    points = np.zeros((2,2,3))
    output = np.array([[0, 0, 1], [width, 0, 1], [0, height, 1], [width, height, 1]]).reshape(2, 2, 3)
    points = np.tensordot(output, np.linalg.inv(K).T, axes = 1)
    points = depth * points
    points = np.tensordot(points, Rt[0:3, 0:3], axes = 1)
    points = points - Rt[:3, :3].T.dot(Rt[:3, 3])
    return points


def preprocess_ncc_impl(image, ncc_size):
    """
    Prepare normalized patch vectors according to normalized cross
    correlation.

    This is a preprocessing step for the NCC pipeline.  It is expected that
    'preprocess_ncc' is called on every input image to preprocess the NCC
    vectors and then 'compute_ncc' is called to compute the dot product
    between these vectors in two images.

    NCC preprocessing has two steps.
    (1) Compute and subtract the mean.
    (2) Normalize the vector.

    The mean is per channel.  i.e. For an RGB image, over the ncc_size**2
    patch, compute the R, G, and B means separately.  The normalization
    is over all channels.  i.e. For an RGB image, after subtracting out the
    RGB mean, compute the norm over the entire (ncc_size**2 * channels)
    vector and divide.

    If the norm of the vector is < 1e-6, then set the entire vector for that
    patch to zero.

    Patches that extend past the boundary of the input image at all should be
    considered zero.  Their entire vector should be set to 0.

    Patches are to be flattened into vectors with the default numpy row
    major order.  For example, given the following
    2 (height) x 2 (width) x 2 (channels) patch, here is how the output
    vector should be arranged.

    channel1         channel2
    +------+------+  +------+------+ height
    | x111 | x121 |  | x112 | x122 |  |
    +------+------+  +------+------+  |
    | x211 | x221 |  | x212 | x222 |  |
    +------+------+  +------+------+  v
    width ------->

    v = [ x111, x121, x211, x221, x112, x122, x212, x222 ]

    Input:
        image -- height x width x channels image of type float32
        ncc_size -- integer width and height of NCC patch region.
    Output:
        normalized -- heigth x width x (channels * ncc_size**2) array
    """
    h, w, c = image.shape
    normalized = np.zeros((h,w,c*ncc_size**2))
    image_copy = np.copy(image)
    size = ncc_size//2
    for i in range(size, h-size):
        for j in range(size, w - size):
            patch = image_copy[i - size:i + size+1, j - size:j + size+1,:]
            patch_mean = np.mean(patch, axis = (0, 1)) # subtract mean
            patch = patch - patch_mean
            patch_flat = patch.reshape(ncc_size**2, c).T.flatten() # flatten
            norm = np.linalg.norm(patch_flat)
            if norm < 1e-6: # check norm
                patch = np.zeros((c * ncc_size**2))
            else: 
                patch = patch_flat/norm
            normalized[i,j] = patch
    return normalized


def compute_ncc_impl(image1, image2):
    """
    Compute normalized cross correlation between two images that already have
    normalized vectors computed for each pixel with preprocess_ncc.

    Input:
        image1 -- height x width x (channels * ncc_size**2) array
        image2 -- height x width x (channels * ncc_size**2) array
    Output:
        ncc -- height x width normalized cross correlation between image1 and
               image2.
    """
    ncc = np.sum(image1 * image2, axis = 2)
    return ncc
