import os
import sys
import gradio as gr
from gradio.components import Dropdown, CheckboxGroup, Textbox, Video, Image
import sdapi
sys.path.insert(1, ".\\anime-segmentation-main\\")
sys.path.insert(1, ".\\talking-head-anime-2-demo-simpleversion\\")
import anime_bgrm
import face_script

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

def clearDir():
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

def generate_gifs(text_input, skin_color, eye_color, hair_color, hair_style, accessories, video_input):
    clearDir()
    os.chdir(main_dir)
    text_input = "white background, " + text_input
    if skin_color != "":
        skin_color = skin_color + " skin, "
    if eye_color != "":
        eye_color = eye_color + " eyes, "
    if hair_color != "": 
        hair_color = hair_color + " hair, "
    if hair_style != "":
        hair_style = hair_style + " hair, "
    accessories_text = ""
    for i in accessories:
        accessories_text = accessories_text + i + ", "
    generate_imgs(text_input + ", " + skin_color + eye_color + hair_color + accessories_text)
    os.chdir(seg_script_dir)
    anime_bgrm.main(data=seg_input_dir, out=seg_output_dir)
    os.chdir(vid_script_dir)
    return face_script.main(vid_input_dir, video_input, vid_output_dir)

# def load_model(path, net_name, img_size):
#     # 如果需要load model(stable diffusion, anime_rmbg),参考sd modelloader.py, webui.py，anime anime_bgrm.py
#     # return "success"

# def get_model_path():
#     # get model path
#     return model_path

def generate_imgs(text_input):
    return sdapi.main("output/stable-diffusion/", 2, text_input)

app = gr.Blocks(css="./css.css")
with app:
    # text input that has cap of 75 characters
    text_input = Textbox(lines=3, placeholder="Type Enter your text (max 75 characters)...")
    
    # checkbox group
    accessory_options = ["Earrings", "Hair Ornament", "Necklace"]
    accessories = CheckboxGroup(accessory_options, label="Accessories")
    
    # dropdown menus,list placeholders for change:
    skin_colors = ["Light", "Medium", "Dark"]
    eye_colors = ["Blue", "Brown", "Green", "Hazel", "Gray"]
    hair_colors = ["Blonde", "Brunette", "Red", "Black", "Gray", "Auburn", "White"]
    hair_styles = ["Short", "Medium", "Long", "Curly", "Straight"]
    
    with gr.Row():
        skin_color = Dropdown(skin_colors, label="Skin Color")
        eye_color = Dropdown(eye_colors, label="Eye Color")
    
    with gr.Row():
        hair_color = Dropdown(hair_colors, label="Hair Color")
        hair_style = Dropdown(hair_styles, label="Hair Style")

    # input video and output generated GIFs
    with gr.Row():
        video_input = Video(elem_classes="video-preview")
        # define outputs of images
        output_gifs = gr.Gallery(label="Generated Gifs")
    btn = gr.Button(value="Generate", variant="primary")

    # btn.click(generate_imgs, [text_input], [output_gifs])

    # # !!! button click output, use defined functions (fn is placeholder)
    btn.click(generate_gifs, [text_input, skin_color, eye_color, hair_color, hair_style, accessories, video_input],
              [output_gifs])

app.launch()