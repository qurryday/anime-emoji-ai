import os
import argparse

main_dir = os.getcwd()
sd_script_dir = main_dir + "\stable-diffusion-api\\"
sd_output_dir = main_dir + "\output\stable-diffusion\\"

seg_script_dir = main_dir + "\\anime-segmentation-main\\"
seg_input_dir = sd_output_dir
seg_output_dir = main_dir + "\output\\anime-segmentation\\"

vid_script_dir = main_dir + "\\talking-head-anime-2-demo-simpleversion\\"
vid_input_dir = seg_output_dir
vid_output_dir = main_dir + "\output\\talking-head-anime\\"

def remove(dir):
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

def checkDir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def initDir():
    # Make sure output dir exists
    checkDir(sd_script_dir)
    checkDir(sd_output_dir)
    checkDir(seg_script_dir)
    checkDir(seg_output_dir)
    checkDir(vid_script_dir)
    checkDir(vid_output_dir)

    # Clear output directories
    remove(sd_output_dir)
    remove(seg_output_dir)
    remove(vid_output_dir)

if __name__ == "__main__":
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=1, help='batch size: 1~4')
    opt = parser.parse_args()

    initDir()

    # Execute stable diffusion api
    os.chdir(sd_script_dir)
    os.system("python stable-diffusion-api.py --out " + sd_output_dir + " --size " + str(opt.size))

    # Execute anime segmentation: remove background
    os.chdir(seg_script_dir)
    os.system("python anime_bgrm.py --data " + seg_input_dir + " --out " + seg_output_dir)

    # Execute video generation
    os.chdir(vid_script_dir)
    os.system("python my_script.py --input " + vid_input_dir + " --out " + vid_output_dir)
    

