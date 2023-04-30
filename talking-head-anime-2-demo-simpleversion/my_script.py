import cv2
import os
import PIL.Image
import io
from io import StringIO, BytesIO
import IPython.display
import numpy
import ipywidgets
from tha2.util import extract_PIL_image_from_filelike, resize_PIL_image, extract_pytorch_image_from_PIL_image, convert_output_image_from_torch_to_numpy
import tha2.poser.modes.mode_20
import time
import threading
import torch
import argparse

FRAME_RATE = 30.0
DEVICE_NAME = 'cuda'
device = torch.device(DEVICE_NAME)

input_image_name = "./input/5.png"
output_video_name = 'test_output_video.mp4'

images = []

poser = tha2.poser.modes.mode_20.create_poser(device)
pose_parameters = tha2.poser.modes.mode_20.get_pose_parameters()
pose_size = poser.get_num_parameters()
last_pose = torch.zeros(1, pose_size).to(device)

iris_small_left_index = pose_parameters.get_parameter_index("iris_small_left")
iris_small_right_index = pose_parameters.get_parameter_index("iris_small_right")
iris_rotation_x_index = pose_parameters.get_parameter_index("iris_rotation_x")
iris_rotation_y_index = pose_parameters.get_parameter_index("iris_rotation_y")
head_x_index = pose_parameters.get_parameter_index("head_x")
head_y_index = pose_parameters.get_parameter_index("head_y")
neck_z_index = pose_parameters.get_parameter_index("neck_z")

p = [[0,0,0,0,0,0,0,0,0,0,0,0,0,-0.1,0,0],
     [0,0,0.1,0,0,0,0,0,0,0,0,0,-0.2,0,0],
     [0,0,0.2,0,0,0,0,0,0,0,0,0,-0.25,0,0],
     [0,0,0.3,0,0,0,0,0,0,0,0,0,-0.3,0,0],
     [0,0,0.4,0,0,0,0,0,0,0,0,0,-0.35,0,0],
     [0,0,0.5,0,0,0,0,0,0,0,0,0,-0.4,0,0],
     [0,0,0.6,0,0,0,0,0,0,0,0,0,-0.45,0,0],
     [0,0,0.7,0,0,0,0,0,0,0,0,0,-0.5,0,0],
     [0,0,0.8,0,0,0,0,0,0,0,0,0,-0.5,0,0],
     [0,0,0.9,0,0,0,0,0,0,0,0,0,-0.5,0,0],
     [0,0,1,0,0,0,0,0,0,0,0,0,-0.5,0,0],
     [0,0,1,0,0,0,0,0,0,0,0,0,-0.5,0,0],
     [0,0,1,0,0,0,0,0,0,0,0,0,-0.5,0,0],
     [0,0,1,0,0,0,0,0,0,0,0,0,-0.5,0,0],
     [0,0,1,0,0,0,0,0,0,0,0,0,-0.5,0,0],
     [0,0,1,0,0,0,0,0,0,0,0,0,-0.5,0,0],
     [0,0,1,0,0,0,0,0,0,0,0,0,-0.5,0,0],
     [0,0,1,0,0,0,0,0,0,0,0,0,-0.5,0,0],
     [0,0,1,0,0,0,0,0,0,0,0,0,-0.5,0,0],
     [0,0,0.8,0,0,0,0,0,0,0,0,0,-0.5,0,0],
     [0,0,0.6,0,0,0,0,0,0,0,0,0,-0.5,0,0],
     [0,0,0.4,0,0,0,0,0,0,0,0,0,-0.5,0,0],
     [0,0,0.2,0,0,0,0,0,0,0,0,0,-0.45,0,0],
     [0,0,0,0,0,0,0,0,0,0,0,0,-0.4,0,0],
     [0,0,0,0,0,0,0,0,0,0,0,0,-0.3,0,0],
     [0,0,0,0,0,0,0,0,0,0,0,0,-0.2,0,0],
     [0,0,0,0,0,0,0,0,0,0,0,0,-0.1,0,0],
     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],]

def get_pose(ii):
    pose = torch.zeros(1, pose_size)

    eyebrow_name = "troubled"
    eyebrow_name = f"eyebrow_{eyebrow_name}"
    eyebrow_left_index = pose_parameters.get_parameter_index(f"{eyebrow_name}_left")
    eyebrow_right_index = pose_parameters.get_parameter_index(f"{eyebrow_name}_right")
    pose[0, eyebrow_left_index] = p[ii][0]
    pose[0, eyebrow_right_index] = p[ii][1]

    eye_name = "wink"
    eye_name = f"eye_{eye_name}"
    eye_left_index = pose_parameters.get_parameter_index(f"{eye_name}_left")
    eye_right_index = pose_parameters.get_parameter_index(f"{eye_name}_right")
    pose[0, eye_left_index] = p[ii][2]
    pose[0, eye_right_index] = p[ii][3]

    mouth_name = "aaa"
    mouth_name = f"mouth_{mouth_name}"
    if mouth_name == "mouth_lowered_corner" or mouth_name == "mouth_raised_corner":
        mouth_left_index = pose_parameters.get_parameter_index(f"{mouth_name}_left")
        mouth_right_index = pose_parameters.get_parameter_index(f"{mouth_name}_right")
        pose[0, mouth_left_index] = p[ii][4]
        pose[0, mouth_right_index] = p[ii][5]
    else:
        mouth_index = pose_parameters.get_parameter_index(mouth_name)
        pose[0, mouth_index] = p[ii][4]

    pose[0, iris_small_left_index] = p[ii][6]
    pose[0, iris_small_right_index] = p[ii][7]
    pose[0, iris_rotation_x_index] = p[ii][8]
    pose[0, iris_rotation_y_index] = p[ii][9]
    pose[0, head_x_index] = p[ii][10]
    pose[0, head_y_index] = p[ii][11]
    pose[0, neck_z_index] = p[ii][12]

    return pose.to(device)

def main(inputdir, outputdir):
    i = 0
    for f in os.listdir(inputdir):
        generate_vid(inputdir + f, outputdir + "output_" + str(i) + ".mp4")
        i = i + 1

def generate_vid(input, output):
    global torch_input_image
    pil_image = resize_PIL_image(extract_PIL_image_from_filelike(input))
    w, h = pil_image.size
    if pil_image.mode != 'RGBA':
        AssertionError("Image must have an alpha channel!!!")
    else:
        torch_input_image = extract_pytorch_image_from_PIL_image(pil_image).to(device)
        output_image = torch_input_image.detach().cpu()
        numpy_image = numpy.uint8(numpy.rint(convert_output_image_from_torch_to_numpy(output_image) * 255.0))
        # pil_image = PIL.Image.fromarray(numpy_image, mode='RGBA')
        # IPython.display.display(pil_image)

        images = []
        for i in range(len(p)):
            pose = get_pose(i)
            pytorch_image = poser.pose(torch_input_image, pose)[0]
            output_image = pytorch_image.detach().cpu()
            numpy_image = numpy.uint8(numpy.rint(convert_output_image_from_torch_to_numpy(output_image) * 255.0))
            # pil_image = PIL.Image.fromarray(numpy_image, mode='RGBA')
            im = PIL.Image.fromarray(numpy_image)
            im.save("./images/myIm" + str(i) + ".png")

        import moviepy.video.io.ImageSequenceClip
        import os
        image_folder='./images'
        fps=30

        image_files = [os.path.join(image_folder,img)
                    for img in os.listdir(image_folder)
                    if img.endswith(".png")]
        
        image_files.sort(key=lambda x:int((x.split('myIm')[1]).split('.')[0]))
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files[:len(p)], fps=fps)
        clip.write_videofile(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model args
    parser.add_argument('--input', type=str, default=input_image_name, help='input dir')
    parser.add_argument('--out', type=str, default=output_video_name, help='output dir')
    opt = parser.parse_args()
    main(opt.input, opt.out)
