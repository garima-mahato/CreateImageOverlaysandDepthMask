import os
from random import random
from random import seed

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from CreateDepthMaskNYU import CreateDepthMaskNYU

seed(2)
model_path = r'/home/abhijit/PycharmProjects/Test/NYU/nyu.h5'
depth_model = CreateDepthMaskNYU(model_path)

bg_path = r'/media/abhijit/DATA/Development/TSAI/EVA/MaskRCNN Dataset/Background'
fg_path = r'/media/abhijit/DATA/Development/TSAI/EVA/MaskRCNN Dataset/Foreground'
fg_mask_path = r'/media/abhijit/DATA/Development/TSAI/EVA/MaskRCNN Dataset/Foreground/masks'

final_output = r'/media/abhijit/DATA/Development/TSAI/EVA/MaskRCNN Dataset/OverLayedImages'
final_output_mask = r'/media/abhijit/DATA/Development/TSAI/EVA/MaskRCNN Dataset/OverLayedMask'
final_output_dm = r'/media/abhijit/DATA/Development/TSAI/EVA/MaskRCNN Dataset/OverLayedDepthMasks'


fg_image_ext = '.png'
fg_image_flip_ext = '_flip.png'
bg_image_ext = '.jpg'
final_image_ext = '.jpg'
final_output_dm_ext = '.jpg'
batch_images_count = 40000
batch = 1
batch_folder_name = 'batch_'

for x in tqdm(range(0, 40)):

    init_x = int(np.floor(random() * 79))
    init_y = int(np.floor(random() * 79))
    dict_for_dm = {}

    # if not x == 0 and x % 4 == 0:
    #     batch += 1;

    if not x == 0:
        batch += 1;

    fg_img_path = final_output + os.path.sep + batch_folder_name + str(batch)
    if not os.path.exists(fg_img_path):
        os.mkdir(fg_img_path)

    fg_mask_img_path = final_output_mask + os.path.sep + batch_folder_name + str(batch)
    if not os.path.exists(fg_mask_img_path):
        os.mkdir(fg_mask_img_path)

    fg_dm_img_path = final_output_dm + os.path.sep + batch_folder_name + str(batch)
    if not os.path.exists(fg_dm_img_path):
        os.mkdir(fg_dm_img_path)

    for mx in tqdm(range(1, 101)):

        # if len(dict_for_dm > 0):
        #     depth_model.flush_depth_maps(dict_for_dm)
        #     dict_for_dm = {}

        overlay_file_path = "{0}{1}{2}{3}".format(fg_path, os.path.sep, str(mx), fg_image_ext)
        overlay_file_mask_path = "{0}{1}{2}{3}".format(fg_mask_path, os.path.sep, str(mx), fg_image_ext)

        overlay = cv2.imread(overlay_file_path, cv2.IMREAD_UNCHANGED)
        overlay_mask = cv2.imread(overlay_file_mask_path, cv2.IMREAD_UNCHANGED)

        flip_overlay_path = "{0}{1}{2}{3}".format(fg_path, os.path.sep, str(mx), fg_image_flip_ext)
        flip_overlay_mask_path = "{0}{1}{2}{3}".format(fg_mask_path, os.path.sep, str(mx), fg_image_flip_ext)

        flip_overlay = cv2.imread(flip_overlay_path, cv2.IMREAD_UNCHANGED)
        flip_overlay_mask = cv2.imread(flip_overlay_mask_path, cv2.IMREAD_UNCHANGED)

        for k in (range(1, 101)):
            bg_file_path = "{0}{1}{2}{3}".format(bg_path, os.path.sep, str(k), bg_image_ext)
            background = cv2.imread(bg_file_path, cv2.IMREAD_UNCHANGED)
            empty_image = np.zeros((224, 224, 3), dtype="uint8")

            for i in range(init_x, init_x + overlay.shape[0]):
                for j in range(init_y, init_y + overlay.shape[1]):
                    if x < 20:
                        if overlay[i - init_x, j - init_y, 3] != 0:
                            background[i, j, 0:3] = overlay[i - init_x, j - init_y, 0:3]
                    else:
                        if flip_overlay[i - init_x, j - init_y, 3] != 0:
                            background[i, j, 0:3] = flip_overlay[i - init_x, j - init_y, 0:3]

            for ix in range(0, 3):
                if x < 20:
                    empty_image[init_x:init_x + overlay.shape[0], init_y:init_y + overlay.shape[1], ix] = overlay_mask[
                                                                                                          0:
                                                                                                          overlay.shape[
                                                                                                              0],
                                                                                                          0:
                                                                                                          overlay.shape[
                                                                                                              1]]
                else:
                    empty_image[init_x:init_x + overlay.shape[0], init_y:init_y + overlay.shape[1],
                    ix] = flip_overlay_mask[
                          0:
                          overlay.shape[
                              0],
                          0:
                          overlay.shape[
                              1]]

            final_image_path = fg_img_path + os.path.sep + str(k) + '_p' + str(mx) + '_it' + str(x) + final_image_ext
            cv2.imwrite(final_image_path, background)

            final_image_mask_path = fg_mask_img_path + os.path.sep + str(k) + '_p' + str(mx) + '_it' + str(x) \
                                    + final_image_ext

            empty_image = cv2.cvtColor(empty_image, cv2.COLOR_BGR2GRAY)

            cv2.imwrite(final_image_mask_path, empty_image)

            final_image_dm_path = fg_dm_img_path + os.path.sep + str(k) + '_p' + str(mx) + '_it' + str(
                x) + final_output_dm_ext

            dict_for_dm[final_image_dm_path] = background

            width = background.shape[1] * 2
            height = background.shape[0] * 2
            dim = (width, height)

            # resize image
            resized = cv2.resize(background, dim, interpolation=cv2.INTER_AREA)

            cv2.imwrite('/home/abhijit/PycharmProjects/Test/temp.png', resized)

            dm = depth_model.get_depth_map(background, '/home/abhijit/PycharmProjects/Test/temp.png', 1)

            depth_model.save_dm(dm, final_image_dm_path)
