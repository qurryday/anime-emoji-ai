# AnimeMoji: Animated Anime Emoji

This project is an integration of three networks:
1. [Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
2. [Anime Segmentation](https://github.com/SkyTNT/anime-segmentation/)
3. [Talking Head Anime](https://github.com/pkhungurn/talking-head-anime-2-demo)

#### Before running
To run this project, Stable Diffusion needs to be installed and running. Please check the link above for installation guides. The ckpt model we are using is in the link below.

Add --api as the commandline argument in webui-user.bat in your Stable Diffusion folder:
```
@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=--api

call webui.bat
```

For other environment requirements, please check requirements.txt.

#### How to run

Run Stable Diffusion Web UI by webui-user.bat.

Then, run app.py from termial.

#### More information:
https://www.calvinzqiu.com/animoji

#### Helpful links: 

[Google Drive Folder](https://drive.google.com/drive/folders/1eJQkC4WhZyMDjB0IBZWSqvKjO0KHLaP4?usp=sharing)

[Model: momoke-e.ckpt](https://huggingface.co/LarryAIDraw/momoko-e/tree/main)
