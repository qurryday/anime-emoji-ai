# Reference: https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API

import json
import requests
import io
import base64
import time
import argparse
from PIL import Image, PngImagePlugin

def main(output_dir, batch_size, user_prompt=""):
    url = "http://127.0.0.1:7860"

    if user_prompt == "":
        user_prompt = input("Please enter your prompt: ")

    payload = {
        "sd_model_checkpoint": "momoke-e.ckpt [18bcc837a2]",
        "prompt": user_prompt,
        "styles": ["general", "avatar"],
        "steps": 20, 
        "batch_size": batch_size
    }

    response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)

    r = response.json()

    count = 0

    imglist = []

    for i in r['images']:
        image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))

        png_payload = {
            "image": "data:image/png;base64," + i
        }
        response2 = requests.post(url=f'{url}/sdapi/v1/png-info', json=png_payload)

        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("parameters", response2.json().get("info"))
        image_name = output_dir + time.strftime("%Y-%m-%dT%H-%M-%S") + "_" + str(count) + ".png"
        image.save(image_name, pnginfo=pnginfo)
        print("Saved as " + image_name)
        count = count + 1
        imglist = imglist + [image]
    return imglist

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model args
    parser.add_argument('--out', type=str, default='output/', help='output dir')
    parser.add_argument('--size', type=int, default=1, help='batch size: 1~4')
    opt = parser.parse_args()
    if opt.size < 1 or opt.size > 4:
        AssertionError("Batch size should be 1 ~ 4")
    main(opt.out, opt.size)