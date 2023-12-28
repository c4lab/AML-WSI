# This script gets a tile and returns detected and classified cells inside and saves it. script is for v2.x.x. of YOLOv4
import cv2
import sys
import os
import numpy as np
import tensorflow as tf
import argparse
import pickle
import time
from yolov4.tf import YOLOv4
# import argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_patch", required=True, default=None, type=str, help="input patch file, should be .png")
    parser.add_argument("--model_weights", required=False, default="./model/yolo-obj_best.weights", type=str, help="model weights file")
    parser.add_argument("--object_names", required=False, default="./model/obj.names", type=str, help="model object class names file")
    parser.add_argument("--out_dir", required=False, default="./", type=str, help="output directory")
    parser.add_argument("--cell_size", required=False, default=64, type=int, help="cell size")
    args = parser.parse_args()
    return args
# function to crop the cells
def crop_cells(image_name,image,bboxes,classes,out_dir,cell_size):
    num_boxes = bboxes
    image_h, image_w, color = image.shape
    for i in range(num_boxes.shape[0]):
        start_x = int(num_boxes[i][0] * image_w)
        start_y = int(num_boxes[i][1] * image_h)
        cell_w = int(num_boxes[i][2] * image_w)
        cell_h = int(num_boxes[i][3] * image_h)
        if os.path.exists(out_dir+f"{classes[int(num_boxes[i][4])]}") == False:
            os.makedirs(out_dir+f"{classes[int(num_boxes[i][4])]}")
        # crop the cell depends on the larger edge
        if cell_w > cell_h:
            sub_image = image[start_y-cell_w//2:start_y+cell_w//2 , start_x-cell_w//2:start_x+cell_w//2]
        else:
            sub_image = image[start_y-cell_h//2:start_y+cell_h//2 , start_x-cell_h//2:start_x+cell_h//2]
        if sub_image.shape[0] == 0 or sub_image.shape[1] == 0 or sub_image.shape[2] == 0:
            continue
        # resize the cell to 64*64
        
        sub_image = cv2.resize(sub_image,(cell_size,cell_size))
        # save the cell image
        cv2.imwrite(f"{out_dir+classes[int(num_boxes[i][4])]}/{image_name}_{i+1}.png", sub_image)
def save_bboxes(slide_name,patch_name,pred_bboxes,out_dir):
    # check if the pickle file exists, if not, create one
    if os.path.exists(f"{out_dir}/{slide_name}.pkl") == False:
        #create pickle file
        with open(f"{out_dir}/{slide_name}.pkl", 'wb') as f:
            my_dict = {patch_name:pred_bboxes}
            pickle.dump(my_dict, f)
    else:
        # load the pickle file
        with open(f"{out_dir}/{slide_name}.pkl", 'rb') as f:
            my_dict = pickle.load(f)
        # add the new image to the dictionary
        my_dict[patch_name] = pred_bboxes
        # save the dictionary to the pickle file
        with open(f"{out_dir}/{slide_name}.pkl", 'wb') as f:
            pickle.dump(my_dict, f)
def check_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
def main(args):
    check_gpu() # mutiple gpu with different memory
    yolo = YOLOv4()
    with open(args.object_names, 'r') as f:
        classes = f.read().splitlines()
    yolo.classes = args.object_names
    yolo.make_model()
    yolo.load_weights(args.model_weights, weights_type="yolo")
    original_image = cv2.imread(args.input_patch)
    resized_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    resized_image = yolo.resize_image(resized_image)
    resized_image = resized_image / 255
    input_data = resized_image[np.newaxis, ...].astype(np.float32)
    candidates = yolo.model.predict(input_data)
    _candidates = []
    for candidate in candidates:
        batch_size = candidate.shape[0]
        grid_size = candidate.shape[1]
        _candidates.append(
            tf.reshape(
                candidate, shape=(1, grid_size * grid_size * 3, -1)
            )
        )
    candidates = np.concatenate(_candidates, axis=1)
    pred_bboxes = yolo.candidates_to_pred_bboxes(candidates[0])
    pred_bboxes = yolo.fit_pred_bboxes_to_original(
        pred_bboxes, original_image.shape
    )

    result = yolo.draw_bboxes(original_image, pred_bboxes)
    # save the box image
    input_name = args.input_patch.split("/")[-1]
    patch_name = input_name.split(".")[0]
    slide_name = input_name.split("_")[0]
    # create the slide directory
    if os.path.exists(f"{args.out_dir}/{slide_name}/") == False:
        os.makedirs(f"{args.out_dir}/{slide_name}/")
    output_dir = f"{args.out_dir}/{slide_name}/"
    # save the cropped cells
    cv2.imwrite(f"{output_dir}/{patch_name}.png", result)
    crop_cells(patch_name,original_image, pred_bboxes,classes,output_dir,args.cell_size)
    save_bboxes(slide_name,patch_name,pred_bboxes,output_dir)

            
if __name__ == "__main__":
    start_time = time.time()
    args = get_args()
    print(f"Code is running. please check the log file in {args.out_dir}/output.log")
    sys.stdout = open(args.out_dir+"/output.log", 'w')
    sys.stderr = open(args.out_dir + "/error.log", 'w')
    # run the main function
    main(args)
    exec_time = time.time() - start_time
    print("time: {:02d}m{:02d}s".format(int(exec_time // 60), int(exec_time % 60)))
    
