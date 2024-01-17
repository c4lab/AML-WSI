# By Rohollah Moosavi Tayebi, email: rohollah.moosavi@uwaterloo.ca/moosavi.tayebi@gmail.com

import os
os.environ['PATH'] = "C:/openslide-win64/bin" + ";" + os.environ['PATH']
import openslide
import numpy as np
import time
from production_result_module import real_result
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

def tile_selection_module(ROI_path, Tile_Text_File, tile_size, down_scale, threshold, model_name, grid_size, wsi_path, weight_path):
    op = openslide.OpenSlide(wsi_path)
    meda_data = op.properties

    num_pixel_height = int(meda_data['openslide.level[0].height'])
    num_pixel_width = int(meda_data['openslide.level[0].width'])
    print(f'Height of WSI: {num_pixel_height}')
    print(f'Width of WSI: {num_pixel_width}')
    print(f"grid size = {grid_size}")
    print(f'Number of total tested tiles: {grid_size[0]} * {grid_size[1]} = {grid_size[0]*grid_size[1]}')
    print(meda_data)

    start_time = time.time()
    if num_pixel_height > num_pixel_width:
        temp = grid_size[0]
        grid_size[0] = grid_size[1]
        grid_size[1] = temp

    dist_point_width = num_pixel_width//grid_size[0]
    dist_point_height = num_pixel_height//grid_size[1]
    pos_height = dist_point_height
    pos_width = dist_point_width
    num = 1

    image_list = []
    while pos_height < num_pixel_height:
        while pos_width < num_pixel_width:
            img = op.read_region((pos_width, pos_height), 0, (tile_size, tile_size))
            img = img.convert('RGB')
            img = Image.fromarray(np.uint8(img))
            image_list.append(img)
            pos_width += dist_point_width
            num += 1
        pos_height += dist_point_height
        pos_width = dist_point_width

    real_result(tile_size, image_list, down_scale, model_name, threshold, ROI_path, weight_path)

    image_files = []
    os.chdir(os.path.join(ROI_path, ""))
    for filename in os.listdir(os.getcwd()):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_files.append(ROI_path + '/' + filename)

    shuffle_idx = np.random.permutation(len(image_files))
    image_files = [image_files[i] for i in shuffle_idx]

    os.chdir("..")
    with open(ROI_path + "/" + Tile_Text_File, "w") as outfile:
        for image in image_files:
            outfile.write(image)
            outfile.write("\n")
        outfile.close()
    os.chdir("..")

    end_time = time.time()
    exec_time = end_time - start_time
    print("time: {:.2f} ms".format(exec_time * 1000))
