import glob
import os

import matplotlib.pyplot as plt
from PIL import Image
import NYU
from NYU import BilinearUpSampling2D, predict, scale_up, display_images, load_images


class CreateDepthMaskNYU:

    def __init__(self, model_path):
        self.model = None

        if os.path.isfile(model_path):
            print('Loading model...')

            # Custom object needed for inference and training
            custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

            # Load model into GPU / CPU
            self.model = NYU.load_model(model_path, custom_objects=custom_objects, compile=False)

            print('\nModel loaded ({0}).'.format(model_path))

    def get_depth_map(self, image, image_path, scale):

        if self.model is not None:
            # Input images

            inputs = load_images(glob.glob(image_path))

            if scale > 1:
                inputs = scale_up(scale, inputs)

            # Compute results
            outputs = predict(self.model, inputs)
            # outputs = scale_up((1 / (scale / 2)), outputs)
            image = display_images(outputs.copy())
            # image = scale_image(shape=(224, 224), image=image)
            return image

    def save_dm(self, image, image_path):
        fig = plt.figure(figsize=(2.24, 2.24))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(image, aspect='equal')
        plt.savefig(image_path, dpi=100)
        ax.cla()
        ax.clear()
        plt.close(fig)

    def save_dm_160(self, image, image_path):
        fig = plt.figure(figsize=(1.60, 1.60))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(image, aspect='equal')
        plt.savefig(image_path, dpi=100)
        ax.cla()
        ax.clear()
        plt.close(fig)
