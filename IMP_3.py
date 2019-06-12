import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy.ndimage.filters import convolve
from scipy.ndimage import convolve as con


import os

CONV_MAT = np.array([1, 1])
GRASCALE_REPRE = 1
RGB_REPRE = 2
COLOR_LEVEL = 256


def relpath(filename):

    '''
    A function that was given by the school.
    '''

    return os.path.join(os.path.dirname(__file__), filename)

def read_image(filename, representation):

    '''
    A function that converts the image to a desired representation, and with
    intesities normalized to the range of [0,1]
    :param filename: the filename of an image on disk, could be grayscale or
    RGB
    :param representation: representation code, either 1 or 2 defining whether
    the output should be a grayscale image (1) or an RGB image (2)
    :return: an image in the desired representation.
    '''

    im = imread(filename)
    if representation == GRASCALE_REPRE:
        return rgb2gray(im)
    im_float = im.astype(np.float64)
    im_float /= (COLOR_LEVEL - 1)
    return im_float


def build_gaussian_pyramid(im, max_levels, filter_size):

    '''
    Function that construct a Gaussian pyramid of a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the Gaussian filter (an odd scalar that
    represents a squared filter) to be used in constructing the pyramid filter
    :return:
    filter_vec - row vector of shape (1, filter_size) used for the pyramid
    construction
    pyr - a standard python array with maximum length of max_levels, where each
    element of the array is a grayscale image.
    '''

    filter_vec = np.array([1, 1])
    pyr = [im]
    im_n = im
    for i in range(filter_size-2):
        filter_vec = np.convolve(filter_vec, CONV_MAT)
    filter_vec = (1 / np.sum(filter_vec)) * filter_vec[None,:]


    while (len(pyr) < max_levels and np.shape(pyr[-1])[0] / 2 >= 16 and
                       np.shape(pyr[-1])[1] / 2 >= 16):
        im_n = convolve(im_n, filter_vec, mode='nearest')
        im_n = (convolve(im_n.transpose(), filter_vec, mode='nearest')).transpose()
        im_n = im_n[::2, ::2]
        pyr.append(im_n)

    return pyr, filter_vec


def build_laplacian_pyramid(im, max_levels, filter_size):

    '''
    Function that construct a Laplacian pyramid of a given image.
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the Gaussian filter (an odd scalar that
    represents a squared filter) to be used in constructing the pyramid filter
    :return:
    filter_vec - row vector of shape (1, filter_size) used for the pyramid
    construction
    pyr - a standard python array with maximum length of max_levels, where each
    element of the array is a grayscale image.
    '''

    filter_vec = np.array([1, 1])
    pyr = [im]
    im_n = im
    for i in range(filter_size-2):
        filter_vec = np.convolve(filter_vec, CONV_MAT)
    filter_vec = (1 / np.sum(filter_vec)) * filter_vec[None,:]

    while (len(pyr) < max_levels and np.shape(pyr[-1])[0] / 2 > 16 and
                       np.shape(pyr[-1])[1] / 2 > 16):
        orig_im = np.copy(im_n)

        im_n = convolve(im_n, filter_vec)
        im_n = (convolve(im_n.transpose(), filter_vec)).transpose()

        im_n = im_n[::2, ::2]

        index_row = np.arange(1, np.shape(im_n)[0] + 1)
        index_col = np.arange(1, np.shape(im_n)[1] + 1)

        exp_im = np.insert(im_n, index_row, 0, axis=0)
        exp_im = np.insert(exp_im, index_col, 0, axis=1)

        exp_im = convolve(exp_im, 2 * filter_vec)
        exp_im = (convolve(exp_im.transpose(), 2 * filter_vec)).transpose()

        pyr.append(orig_im - exp_im)

    pyr = pyr[1:]
    pyr.append(im_n)
    return pyr, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):

    '''
    Reconstruction of an image from its Laplacian Pyramid
    :param lpyr: a standard python array with maximum length of max_levels,
    where each element of the array is a grayscale image.
    :param filter_vec: row vector of shape (1, filter_size) used for the
    pyramid construction
    :param coeff: python list, same as the number of levels in the pyramid
    lpyr. Each level i of the Laplacian pyramid is multiplied by its
    corresponding coefficient
    :return: The reconstructed image from the laplacian array.
    '''
    for i in range(len(coeff)):
        lpyr[i] *= coeff[i]

    res_im = lpyr[-1]

    for i in range(len(lpyr) - 2, -1, -1):

        index_row = np.arange(1, np.shape(res_im)[0] + 1)
        index_col = np.arange(1, np.shape(res_im)[1] + 1)

        res_im = np.insert(res_im, index_row, 0, axis=0)
        res_im = np.insert(res_im, index_col, 0, axis=1)

        res_im = convolve(res_im, 2 * filter_vec)
        res_im = (convolve(res_im.transpose(), 2 * filter_vec)).transpose()

        res_im = res_im + lpyr[i]

    return res_im


def render_pyramid(pyr, levels):

    '''
    function that builds a picture where all pyramid images are displayed.
    :param pyr: either a Gaussian or Laplacian pyramid as defined above.
    :param levels: levels is the number of levels to present in the result.
    :return: res - a single black image in which the pyramid levels of the
    given pyramid pyr are stacked horizontally.
    '''

    width, start = 0, 0
    width = sum(pyr[i].shape[1] for i in range(levels))
    res = np.zeros(shape=(pyr[0].shape[0], width))

    for i in range(levels):
        pyr[i] = (pyr[i] - pyr[i].min())
        pyr[i] = pyr[i] / (pyr[i].max())

    for i in range(levels):
        res[0:pyr[i].shape[0], start:pyr[i].shape[1] + start] = pyr[i]
        start += pyr[i].shape[1]

    return res


def display_pyramid(pyr, levels):

    '''
    function that displays the result of the pyramid at each level in one image.
    :param pyr: either a Gaussian or Laplacian pyramid as defined above.
    :param levels: levels is the number of levels to present in the result.
    :return: res - a single black image in which the pyramid levels of the
    given pyramid pyr are stacked horizontally.
    '''

    plt.imshow(render_pyramid(pyr, levels), cmap='gray')
    plt.show()


def helper_pyramid_blending(im1, im2, mask, max_levels, filter_size_im,
                            filter_size_mask):
    '''
    Helper function in order to avoid double code.
    :param im1: grayscale image to be blended
    :param im2: grayscale image to be blended
    :param mask: a boolean mask containing True and False representing which
     parts of im1 and im2 should appear in the resulting im_blend
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size_im: size of the Gaussian filter used in the construction
    of the Laplacian pyramids of im1 and im2
    :param filter_size_mask: size of the Gaussian filter used in the
    construction of the Gaussian pyramid of mask.
    :return: im_blend - The resulting image of the blend between im1 and im2
    '''

    lapl_pyr_im1, filter_vec1 = build_laplacian_pyramid(im1, max_levels,
                                                        filter_size_im)
    lapl_pyr_im2, filter_vec2 = build_laplacian_pyramid(im2, max_levels,
                                                        filter_size_im)
    mask = mask.astype(np.float64)
    gaus_pyr_mask, filter_vec_gau = build_gaussian_pyramid(mask, max_levels,
                                                           filter_size_mask)



    lapl_out = [im1]
    for i in range(len(lapl_pyr_im2)):
        lapl_out.append( gaus_pyr_mask[i] * lapl_pyr_im1[i] + (1 - gaus_pyr_mask[
            i]) * lapl_pyr_im2[i])

    lapl_out = lapl_out[1:]
    return np.clip(
        laplacian_to_image(lapl_out, filter_vec1, np.ones(len(lapl_pyr_im1))),
        0, 1)



def pyramid_blending(im1, im2, mask, max_levels, filter_size_im,
                     filter_size_mask):
    '''
    function that blends two images, creating a new image where it takes from
    im1 a defined part and from im2 a defined part and blends the two.
    :param im1: image to be blended
    :param im2: image to be blended
    :param mask: a boolean mask containing True and False representing which
     parts of im1 and im2 should appear in the resulting im_blend
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size_im: size of the Gaussian filter used in the construction
    of the Laplacian pyramids of im1 and im2
    :param filter_size_mask: size of the Gaussian filter used in the
    construction of the Gaussian pyramid of mask.
    :return: im_blend - The resulting image of the blend between im1 and im2
    '''

    if np.ndim(im1) == 3 and np.ndim(im2) == 3:
        out_im = np.zeros(np.shape(im1))
        im1 = im1.astype(np.float64) / 255
        im2 = im2.astype(np.float64) / 255
        for i in range(3):
            out_im[:, :, i] = helper_pyramid_blending(im1[:, :, i],
                                                      im2[:, :, i], mask,
                                                      max_levels,
                                                      filter_size_im,
                                                      filter_size_mask)

        return out_im

    return helper_pyramid_blending(im1, im2, mask, max_levels, filter_size_im,
                                   filter_size_mask)



def blending_example1():

    '''
    Function that shows the pyramid_blending function on two images that we choose.
    :return:
    im1 - The image we want to blend on
    im2 - The image we want to take the part from
    mask - A mask we use to mark the part we want to take from im2 and blend in
    im1.
    im_blend - The blended image.
    '''

    im1 = imread(relpath('externals/shoe.jpg')).astype(np.float64)
    im2 = imread(relpath('externals/nyc.jpg')).astype(np.float64)
    mask = imread(relpath('externals/mask_s.jpg')).astype(np.float64)
    mask = rgb2gray(mask).astype(np.bool)


    im_blend = pyramid_blending(im1, im2, mask, 8, 5, 5)

    plt.imshow(im_blend, cmap='gray')
    plt.show()

    fig = plt.figure(figsize=(8, 8))
    columns = 2
    rows = 2
    fig.add_subplot(rows, columns, 1)
    plt.imshow(im1)
    fig.add_subplot(rows, columns, 2)
    plt.imshow(im2)
    fig.add_subplot(rows, columns, 3)
    plt.imshow(mask, cmap='gray')
    fig.add_subplot(rows, columns, 4)
    plt.imshow(im_blend)

    plt.show()

    return im1, im2, mask, im_blend


def blending_example2():

    '''
    Function that shows the pyramid_blending function on two images that we choose.
    :return:
    im1 - The image we want to blend on
    im2 - The image we want to take the part from
    mask - A mask we use to mark the part we want to take from im2 and blend in
    im1.
    im_blend - The blended image.
    '''

    im1 = imread(relpath('externals/caprio.jpg'))
    im2 = imread(relpath('externals/hardy.jpg'))
    mask = imread(relpath('externals/mask_c.jpg'))
    mask = rgb2gray(mask).astype(np.bool)

    im_blend = pyramid_blending(im1, im2, mask, 8, 5, 5)

    fig = plt.figure(figsize=(8, 8))
    columns = 2
    rows = 2
    fig.add_subplot(rows, columns, 1)
    plt.imshow(im1)
    fig.add_subplot(rows, columns, 2)
    plt.imshow(im2)
    fig.add_subplot(rows, columns, 3)
    plt.imshow(mask, cmap='gray')
    fig.add_subplot(rows, columns, 4)
    plt.imshow(im_blend)

    plt.show()

    return im1, im2, mask, im_blend

sdfsd,sdf,m,sdjf= blending_example1()
# blending_example2()

print(m)