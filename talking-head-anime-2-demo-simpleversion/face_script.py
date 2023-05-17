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
import numpy as np
import mediapipe as mp
import argparse
import math
from scipy.signal import savgol_filter


FRAME_RATE = 30.0
DEVICE_NAME = 'cuda'
device = torch.device(DEVICE_NAME)


input_video_name = "./test.mp4"

input_image_name = "./input/"
output_gif_name = ""

def main(inputdir, inputvid, outputdir):
    load_vid(inputvid)
    i = 0
    vidlist = []
    for f in os.listdir(inputdir):
        output_gif_name = outputdir + time.strftime("%Y-%m-%dT%H-%M-%S") + "_" + str(i) + ".gif"
        generate_gif(inputdir + f, output_gif_name)
        i = i + 1
        vidlist = vidlist + [output_gif_name]
    return vidlist

def generate_gif(input, output):
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
            background = PIL.Image.new('RGBA', (256, 256), (255, 255, 255, 255))
            im = PIL.Image.fromarray(numpy_image)
            im = PIL.Image.alpha_composite(background, im)
            im.save("./images/myIm" + str(i) + ".png")
            images = images + [im]

        fps = 30
        duration = int (1.0 / fps * 1000)
        images[0].save(output, save_all=True, append_images=images, optimize=False, loop=0, duration=duration)

# manual wink
# p = [[0,0,0,0,0,0,0,0,0,0,0,0,0,-0.1,0,0],
#      [0,0,0.1,0,0,0,0,0,0,0,0,0,-0.2,0,0],
#      [0,0,0.2,0,0,0,0,0,0,0,0,0,-0.25,0,0],
#      [0,0,0.3,0,0,0,0,0,0,0,0,0,-0.3,0,0],
#      [0,0,0.4,0,0,0,0,0,0,0,0,0,-0.35,0,0],
#      [0,0,0.5,0,0,0,0,0,0,0,0,0,-0.4,0,0],
#      [0,0,0.6,0,0,0,0,0,0,0,0,0,-0.45,0,0],
#      [0,0,0.7,0,0,0,0,0,0,0,0,0,-0.5,0,0],
#      [0,0,0.8,0,0,0,0,0,0,0,0,0,-0.5,0,0],
#      [0,0,0.9,0,0,0,0,0,0,0,0,0,-0.5,0,0],
#      [0,0,1,0,0,0,0,0,0,0,0,0,-0.5,0,0],
#      [0,0,1,0,0,0,0,0,0,0,0,0,-0.5,0,0],
#      [0,0,1,0,0,0,0,0,0,0,0,0,-0.5,0,0],
#      [0,0,1,0,0,0,0,0,0,0,0,0,-0.5,0,0],
#      [0,0,1,0,0,0,0,0,0,0,0,0,-0.5,0,0],
#      [0,0,1,0,0,0,0,0,0,0,0,0,-0.5,0,0],
#      [0,0,1,0,0,0,0,0,0,0,0,0,-0.5,0,0],
#      [0,0,1,0,0,0,0,0,0,0,0,0,-0.5,0,0],
#      [0,0,1,0,0,0,0,0,0,0,0,0,-0.5,0,0],
#      [0,0,0.8,0,0,0,0,0,0,0,0,0,-0.5,0,0],
#      [0,0,0.6,0,0,0,0,0,0,0,0,0,-0.5,0,0],
#      [0,0,0.4,0,0,0,0,0,0,0,0,0,-0.5,0,0],
#      [0,0,0.2,0,0,0,0,0,0,0,0,0,-0.45,0,0],
#      [0,0,0,0,0,0,0,0,0,0,0,0,-0.4,0,0],
#      [0,0,0,0,0,0,0,0,0,0,0,0,-0.3,0,0],
#      [0,0,0,0,0,0,0,0,0,0,0,0,-0.2,0,0],
#      [0,0,0,0,0,0,0,0,0,0,0,0,-0.1,0,0],
#      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],]

def dist(a,b):
    return math.sqrt(abs(a[0]-b[0])^2+abs(a[1]-b[1])^2)

def load_vid(input_video_name):
    # get p
    cap = cv2.VideoCapture(input_video_name)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh_1 = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        # min_detection_confidence=0.6,
        # min_tracking_confidence=0.6
    )
    face_mesh = mp_face_mesh.FaceMesh()

    landmarks = []
    landmarks_1 = []
    current_frame = 0
    while True:
        # Image
        ret, image = cap.read()
        if ret is not True:
            break
        height, width, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Facial landmarks
        result = face_mesh.process(rgb_image)
        result_1 = face_mesh_1.process(rgb_image)

        indices_1 = [469,470,471,472,468,474,475,476,477,473]
        # From our perspective
        ### FOR right eye
        # 0 - 474 right
        # 1 - 475 up
        # 2 - 476 left
        # 3 - 477 down
        # 4 - 473 center
        ### FOR left eye
        # 5 - 469 right
        # 6 - 470 up
        # 7 - 471 left
        # 8 - 472 down
        # 9 - 468 center
        indices = [160,144,158,153,33,133,385,380,387,373,362,263,12,14,78,308,10,152,19,93,323,159,145,386,374,27,223,52,257,443,282]
        # left and right for eye are from our perspective
        # 0 - left eye top 1
        # 1 - left eye bottom 1
        # 2 - left eye top 2
        # 3 - left eye bottom 2
        # 4 - left eye left
        # 5 - left eye right
        # 6 - right eye top 1
        # 7 - right eye bottom 1
        # 8 - right eye top 2
        # 9 - right eye bottom 2
        # 10 - right eye left
        # 11 - right eye right
        # 12 - mouth top
        # 13 - mouth bottom
        # 14 - mouth left
        # 15 - mouth right
        # 16 - head top
        # 17 - head bottom
        # 18 - nose
        # 19 - face left
        # 20 - face right
        # 21 - left eye top
        # 22 - left eye bottom
        # 23 - right eye top
        # 24 - right eye bottom
        # 25 - left eyebrow min
        # 26 - left eyebrow
        # 27 - left eyebrow max
        # 28 - right eyebrow min
        # 29 - right eyebrow
        # 30 - right eyebrow max

        facial_landmarks = result.multi_face_landmarks[0]
        facial_landmarks_1 = result_1.multi_face_landmarks[0]

        landmarks.append([])
        landmarks_1.append([])
        # for i in range(0, 468):
        for i in indices:
            pt1 = facial_landmarks.landmark[i]
            x = int(pt1.x * width)
            y = int(pt1.y * height)
            landmarks[current_frame].append((x,y))
        for i in indices_1:
            pt1 = facial_landmarks_1.landmark[i]
            x = int(pt1.x * width)
            y = int(pt1.y * height)
            landmarks_1[current_frame].append((x,y))
        #     cv2.circle(image, (x, y), 5, (100, 100, 0), -1)
        # cv2.imshow("Image", image)
        # cv2.imshow("Image", image)
        # cv2.waitKey(1)
        current_frame += 1

    # convert landmarks to p
    global p
    p = []
    # rename from right to left
    # EAR_right = abs(landmarks[0][0][1]-landmarks[0][1][1])/abs(landmarks[0][2][0]-landmarks[0][3][0])
    # EAR_left = abs(landmarks[0][0][1]-landmarks[0][1][1])/abs(landmarks[0][2][0]-landmarks[0][3][0])
    next_p = []


    origin_nose_rate = abs(landmarks[0][16][1]-landmarks[0][18][1])/abs(landmarks[0][17][1]-landmarks[0][18][1])
    # iris right space/whole
    origin_left_iris_rate_y = abs(landmarks[0][5][0]-landmarks_1[0][4][0])/max(abs(landmarks[0][5][0]-landmarks[0][4][0]),0.01)
    # iris down space/whole
    origin_left_iris_rate_x = abs(landmarks[0][22][1]-landmarks_1[0][4][1])/max(abs(landmarks[0][21][1]-landmarks[0][22][1]),0.01)
    origin_right_iris_rate_y = abs(landmarks[0][11][0]-landmarks_1[0][9][0])/max(abs(landmarks[0][11][0]-landmarks[0][10][0]),0.01)
    origin_right_iris_rate_x = abs(landmarks[0][24][1]-landmarks_1[0][9][1])/max(abs(landmarks[0][23][1]-landmarks[0][24][1]),0.01)
    origin_iris_rate_x = (origin_left_iris_rate_x+origin_right_iris_rate_x)/2
    origin_iris_rate_y = (origin_left_iris_rate_y+origin_right_iris_rate_y)/2

    # origin_left_eyebrow_min = landmarks[0][25][1]
    # origin_left_eyebrow_max = landmarks[0][26][1]
    # origin_right_eyebrow_min = landmarks[0][27][1]
    # origin_right_eyebrow_max = landmarks[0][28][1]
    # origin_eyebrow_min = (origin_left_eyebrow_min+origin_right_eyebrow_min)/2
    # origin_eyebrow_max = (origin_left_eyebrow_max+origin_right_eyebrow_max)/2

    cf = 0 # current frame
    for landmark in landmarks:
        eyebrow_left = 0
        eyebrow_right = 0
        # rename from right to left
        # EAR range: (0.1-0.25)
        EAR_left_current = (abs(landmark[6][1]-landmark[7][1])+abs(landmark[8][1]-landmark[9][1]))/2/abs(landmark[10][0]-landmark[11][0])
        # EAR range: (0.31-0.56)
        # EAR_left_current = (dist(landmark[6],landmark[7])+dist(landmark[8],landmark[9]))/2/dist(landmark[10],landmark[11])
        eye_left = 1-min((max(EAR_left_current-0.2,0))/0.05,1)
        # eye_left = 1-(EAR_left_current/EAR_left - 0.4)/0.65
        # eye_left = max(1.4-min(EAR_left_current/EAR_left*0.9,1),0)

        # rename from left to right
        EAR_right_current = (abs(landmark[0][1]-landmark[1][1])+abs(landmark[2][1]-landmark[3][1]))/2/abs(landmark[4][0]-landmark[5][0])
        # EAR_right_current = (dist(landmark[0],landmark[1])+dist(landmark[2],landmark[3]))/2/dist(landmark[4],landmark[5])
        eye_right = 1-min((max(EAR_right_current-0.2,0))/0.05,1)
        # eye_right = max(1-(EAR_right_current-0.1)/0.25,0)
        # eye_right = 1-(EAR_right_current/EAR_right - 0.4)/0.65
        # eye_right = max(1.4-min(EAR_right_current/EAR_right*0.9,1.4),0)
        mouth_left = max(min(abs(landmark[12][1]-landmark[13][1])/abs(landmark[14][0]-landmark[15][0])*2-0.1,1.5),0)
        mouth_right = max(min(abs(landmark[12][1]-landmark[13][1])/abs(landmark[14][0]-landmark[15][0])*2-0.1,1.5),0)
        # print(mouth_left)
        iris_small_left = 0
        iris_small_right = 0

        nose_rate = abs(landmark[16][1]-landmark[18][1])/abs(landmark[17][1]-landmark[18][1])
        head_x = (origin_nose_rate - nose_rate)*2
        head_y = -(min(max(abs(landmark[19][0]-landmark[18][0])/max(abs(landmark[20][0]-landmark[19][0]),0.1),0.25),0.75)*3-1.5)
        head_z = np.rad2deg(np.arctan2(landmark[17][0]-landmark[16][0],abs(landmark[16][1]-landmark[17][1])))/15
        
        # 1 - up, -1 - down
        left_iris_rate_y = abs(landmark[5][0]-landmarks_1[cf][4][0])/max(abs(landmarks[0][5][0]-landmarks[0][4][0]),0.01)
        # iris up space/iris down space
        left_iris_rate_x = abs(landmark[22][1]-landmarks_1[cf][4][1])/max(abs(landmarks[0][21][1]-landmarks[0][22][1]),0.01)
        right_iris_rate_y = abs(landmark[11][0]-landmarks_1[cf][9][0])/max(abs(landmarks[0][11][0]-landmarks[0][10][0]),0.01)
        right_iris_rate_x = abs(landmark[24][1]-landmarks_1[cf][9][1])/max(abs(landmarks[0][23][1]-landmarks[0][24][1]),0.01)
        iris_rate_x = (left_iris_rate_x+right_iris_rate_x)/2
        iris_rate_y = (left_iris_rate_y+right_iris_rate_y)/2

        left_eyebrow_min = landmark[25][1]
        left_eyebrow = landmark[26][1]
        left_eyebrow_max = landmark[27][1]
        right_eyebrow_min = landmark[28][1]
        right_eyebrow = landmark[29][1]
        right_eyebrow_max = landmark[30][1]

        # left_eyebrow_raised = (left_eyebrow-left_eyebrow_min)/(left_eyebrow_max-left_eyebrow_min)
        # right_eyebrow_raised = (right_eyebrow-right_eyebrow_min)/(right_eyebrow_max-right_eyebrow_min)
        # eyebrow_raised = ((left_eyebrow_raised+right_eyebrow_raised)/2-0.2)*4
        eyebrow_raised = 0
        iris_rotation_x = max(min((iris_rate_x - origin_iris_rate_x +head_x+eyebrow_raised)*0.7,1),-1)
        # x = iris_rate_x - origin_iris_rate_x
        # coeffient = 400 * x * x
        # # iris_rotation_x = max(min(((iris_rate_x - origin_iris_rate_x)*coeffient + eyebrow_raised),1),-1)
        # print(iris_rotation_x)

        # 1 - left, -1 - right
        # x = iris_rate_y - origin_iris_rate_y+head_y/3
        x = iris_rate_y
        # coeffient = max(200000 * x * x - 100,0)
        coeffient = 8
        iris_rotation_y = max(min((x-0.5)*coeffient,1),-1)+head_y
        # iris_rotation_y = (iris_rate_y - origin_iris_rate_y)
        # print(iris_rate_y)
        
        p.append([eyebrow_left,eyebrow_right,eye_left,eye_right,mouth_left,mouth_right,iris_small_left,
                iris_small_right,iris_rotation_x,iris_rotation_y,head_x,head_y,head_z])
        next_p.append([])

        cf+=1

    # finish get p
    p = np.array(p)
    next_p = np.array(next_p)
    for i in range(len(p[0])):
        b = p[:,len(p[0])-i-1:len(p[0])-i]
        b = np.array([savgol_filter(b.flatten(),15,3)])
        next_p = np.insert(next_p,0,b.flatten(),axis=1)
    p = next_p

    global poser, pose_parameters, pose_size, last_pose 
    poser = tha2.poser.modes.mode_20.create_poser(device)
    pose_parameters = tha2.poser.modes.mode_20.get_pose_parameters()
    pose_size = poser.get_num_parameters()
    last_pose = torch.zeros(1, pose_size).to(device)

    global iris_small_left_index, iris_small_right_index, iris_rotation_x_index, iris_rotation_y_index
    global head_x_index, head_y_index, neck_z_index
    iris_small_left_index = pose_parameters.get_parameter_index("iris_small_left")
    iris_small_right_index = pose_parameters.get_parameter_index("iris_small_right")
    iris_rotation_x_index = pose_parameters.get_parameter_index("iris_rotation_x")
    iris_rotation_y_index = pose_parameters.get_parameter_index("iris_rotation_y")
    head_x_index = pose_parameters.get_parameter_index("head_x")
    head_y_index = pose_parameters.get_parameter_index("head_y")
    neck_z_index = pose_parameters.get_parameter_index("neck_z")

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model args
    parser.add_argument('--input', type=str, default=input_image_name, help='input dir')
    parser.add_argument('--vid', type=str, default=input_video_name, help='input video dir')
    parser.add_argument('--out', type=str, default=output_gif_name, help='output dir')
    opt = parser.parse_args()
    main(opt.input, opt.vid, opt.out)