import numpy as np
from imageio import imread
from tensorflow.keras.layers import Input, Add, Conv2D, Activation
from tensorflow.keras.models import Model, load_model
from scipy import ndimage
from skimage.color import rgb2gray
import math
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
import utils

GRAY_SCALE_IMG = 1
RGB_DIM = 3


def read_image(filename, representation):
    """

    :param filename: filename of an image on disk (grayscale or RGB)
    :param representation: 1 for grayscale, 2 for RGB
    :return: a image converted into an array of type float64 and
            with normalize intensities in the given representation
    """
    img = imread(filename).astype(np.float64)
    img /= 255
    if representation == GRAY_SCALE_IMG and img.ndim == RGB_DIM:
        img = rgb2gray(img)
    return img


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """

    :param filenames: a list of filenames of clean images
    :param batch_size: the size of the batch of images for each iteration of SGD
    :param corruption_func: a function receiving a numpy's array representation of an
           image as a single argument, and returns a randomly corrupted version of
           the input image.
    :param crop_size:  A tuple (height, width) specifying the crop size of the patches
                        to extract
    :return: data_generator which generates tuples (source_batch, target_batch) where
             each output variable is an array of size (batch_size, height, width,1)
    """
    map = {}
    while True:
        source_batch = np.zeros((batch_size, crop_size[0], crop_size[1], 1))
        target_batch = np.zeros((batch_size, crop_size[0], crop_size[1], 1))
        for i in range(batch_size):

            sample = np.random.choice(filenames)
            if sample not in map:
                img = read_image(sample, 1)
                map[sample] = img
            else:
                img = map[sample]

            # creating a big patch
            patch_loc_x = np.random.randint(0, img.shape[0] - crop_size[0] * 3)
            patch_loc_y = np.random.randint(0, img.shape[1] - crop_size[1] * 3)
            large_patch = img[patch_loc_x: patch_loc_x + crop_size[0] * 3,
                          patch_loc_y: patch_loc_y + crop_size[1] * 3]

            # creating the final regular and corrupted patch
            patch_loc_x = np.random.randint(0, large_patch.shape[0] - crop_size[0])
            patch_loc_y = np.random.randint(0, large_patch.shape[1] - crop_size[1])
            corrupted_patch = corruption_func(large_patch)[
                              patch_loc_x:patch_loc_x + crop_size[0],
                              patch_loc_y: patch_loc_y + crop_size[1]] - 0.5
            patch = large_patch[patch_loc_x: patch_loc_x + crop_size[0],
                    patch_loc_y: patch_loc_y + crop_size[1]] - 0.5
            target_batch[i, :, :, 0] = patch
            source_batch[i, :, :, 0] = corrupted_patch

        yield (source_batch, target_batch)

def resblock(input_tensor, num_channels):
    """
    add a residual block to the symbolic input tensor
    :param input_tensor: a symbolic input tensor
    :param num_channels: number of channel of each convolutional layers
    :return: a symbolic output tensor
    """
    b = Conv2D(num_channels, (3, 3), padding='same')(input_tensor)
    b = Activation('relu')(b)
    b = Conv2D(num_channels, (3, 3), padding='same')(b)
    b = Add()([input_tensor, b])
    b = Activation('relu')(b)
    return b


def build_nn_model(height, width, num_channels, num_res_blocks):
    """

    :param height: input height
    :param width: input width
    :param num_channels: number of channel of each convolutional layers
    :param num_res_blocks: number of residual blocks
    :return: an untrained Keras model
    """
    input = Input(shape=(height, width, 1))
    b = Conv2D(num_channels, (3, 3), padding='same')(input)
    b = Activation('relu')(b)
    for i in range(num_res_blocks):
        b = resblock(b, num_channels)
    b = Conv2D(1, (3, 3), padding='same')(b)
    out = Add()([input, b])
    model = Model(inputs=input, outputs=out)
    return model


def train_model(model, images, corruption_func, batch_size, steps_per_epoch,
                num_epochs, num_valid_samples):
    """

    :param model: a general neural network model for image restoration.
    :param images: a list of file paths pointing to image files.  You should assume
                    these paths are complete, and should append anything to them.
    :param corruption_func: a function receiving a numpy's array representation of an
           image as a single argument, and returns a randomly corrupted version of
           the input image.
    :param batch_size: the size of the batch of examples for each iteration of SGD.
    :param steps_per_epoch: The number of update steps in each epoch.
    :param num_epochs: The number of epochs for which the optimization will run.
    :param num_valid_samples: The number of samples in the validation set to test on
                                after every epoch.
    :return:
    """
    training_set = images[: int(0.8 * len(images))]
    validation_set = images[int(0.8 * len(images)):]
    training_set_gen = load_dataset(training_set, batch_size, corruption_func,
                                    model.input_shape[1:3])
    validation_set_gen = load_dataset(validation_set, batch_size, corruption_func,
                                      model.input_shape[1:3])
    model.compile(loss='mean_squared_error', optimizer = Adam(beta_2=0.9))
    model.fit_generator(training_set_gen, steps_per_epoch=steps_per_epoch,
                        epochs=num_epochs, validation_data=validation_set_gen,
                        validation_steps = num_valid_samples)


def restore_image(corrupted_image, base_model):
    """
    :param corrupted_image: a grayscale image of shape(height, width)and with values
                            in the [0,1] range oftype float64
    :param base_model: a neural network trained to restore small patches (
    :return: the restord image
    """
    a = Input(shape=(corrupted_image.shape[0], corrupted_image.shape[1], 1))
    b = base_model(a)
    new_model = Model(inputs=a, outputs=b)
    restored_image = new_model.predict(
        corrupted_image.reshape(1, corrupted_image.shape[0],
                                corrupted_image.shape[1], 1) - 0.5) + 0.5
    return restored_image.clip(0, 1).reshape(corrupted_image.shape[0],
                                             corrupted_image.shape[1]).astype(np.float64)


def add_gaussian_noise(image, min_sigma, max_sigma):
    """
    :param image: a grayscale image with values in the [0,1] range of type float64.
    :param min_sigma: a non-negative scalar value representing the minimal variance
                        of the gaussian distribution.
    :param max_sigma: a non-negative scalar value larger than or equal to min_sigma,
                    representing the maximal variance of the gaussian distribution.
    :return: new image with noise
    """
    sigma = np.random.uniform(min_sigma, max_sigma)
    noise = np.random.normal(0 ,sigma, image.shape)
    corr_image = noise + image
    corr_image = np.round(corr_image * 255) / 255
    return corr_image.clip(0, 1)


def learn_denoising_model(num_res_blocks=5, quick_mode=False):
    """
    :param num_res_blocks: number of residual blocks
    :param quick_mode: True for testing purpose
    :return: a train model
    """
    images = sol5_utils.images_for_denoising()
    model = build_nn_model(24, 24, 48, num_res_blocks)
    if not quick_mode:
        train_model(model, images, lambda img: add_gaussian_noise(img, 0, 0.2), 100,
                    100, 5, 1000)
    else:
        train_model(model, images, lambda img: add_gaussian_noise(img, 0, 0.2), 10,
                    3, 2, 30)
    return model


def add_motion_blur(image, kernel_size, angle):
    """
    :param image: a grayscale image with values in the [0,1] range of type float64.
    :param kernel_size: an odd integer specifying the size of the kernel
    :param angle: an angle in radians in the range [0,π)
    :return: image with blur (with the given angle)
    """
    kernel = sol5_utils.motion_blur_kernel(kernel_size, angle)
    cor_image = ndimage.filters.convolve(image, kernel)
    return cor_image


def random_motion_blur(image, list_of_kernel_sizes):
    """
    :param image: a grayscale image with values in the [0,1] range of type float64.
    :param list_of_kernel_sizes: a list of odd integers.
    :return: image with blur at a random angle
    """
    angle = np.random.uniform(0, math.pi)
    kernel_size = np.random.choice(list_of_kernel_sizes)
    cor_image = add_motion_blur(image, kernel_size, angle)
    cor_image = np.round(cor_image * 255) / 255
    return cor_image.clip(0, 1)


def learn_deblurring_model(num_res_blocks=5, quick_mode=False):
    """

    :param num_res_blocks: number of residual blocks
    :param quick_mode: True for testing purpose
    :return: a train model
    """
    images = sol5_utils.images_for_deblurring()
    model = build_nn_model(16, 16, 32, num_res_blocks)
    if not quick_mode:
        train_model(model, images, lambda img: random_motion_blur(img, [7]), 100,
                    100, 10, 1000)
    else:
        train_model(model, images, lambda img: random_motion_blur(img, [7]), 10, 3,
                    2, 30)
    return model
