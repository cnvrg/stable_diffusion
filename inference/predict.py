# -- coding: utf-8 --`
import os

# engine
import sys
import pathlib
import json

sys.path.append(str(pathlib.Path(__file__).parent.resolve()))
from stable_diffusion_engine import StableDiffusionEngine

# scheduler
from diffusers import LMSDiscreteScheduler, PNDMScheduler
import cv2
import numpy as np
import base64
import traceback
import magic

np.random.seed(None)

#load the models and schedulers needed to run stable diffusion engine
first = LMSDiscreteScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    tensor_format="np",
)
second = PNDMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    skip_prk_steps=True,
    tensor_format="np",
)
txt_to_img_engine = StableDiffusionEngine(
    model="bes-dev/stable-diffusion-v1-4-openvino",
    scheduler=first,
    tokenizer="openai/clip-vit-large-patch14",
)
img_to_img_engine = StableDiffusionEngine(
    model="bes-dev/stable-diffusion-v1-4-openvino",
    scheduler=second,
    tokenizer="openai/clip-vit-large-patch14",
)


def decode_img(data, name):

    """
    This function takes base64 string called data and a string called name. 
    Data will be converted to an image and saved in the local repository with the 
    name argument as the name of the image file.
    """

    decoded = base64.b64decode(data)
    file_ext = magic.from_buffer(decoded, mime=True).split("/")[-1]
    savepath = f"{name}.{file_ext}"  # decode the input file
    f = open(savepath, "wb")
    f.write(decoded)
    f.close()
    return savepath

def predict(data):
    try:
        prompt = data["prompt"]
        steps = int(data["steps"])
        samples = int(data["samples"])
        init_image = None
        mask_image = None

        #check if the user passed init image or mask image
        try:
            init_image = decode_img(data["init_image"], 'init')
        except KeyError:
            pass
        try:
            mask_image = decode_img(data['mask_image'], 'mask')
        except KeyError:
            pass
        generated_images=[]
        prediction = {}

        #generate as many samples for the same prompt as request by the user
        for iteration in range(samples):
            #if the user provided wither init or mask or both images along with prompt use the txt_to_img scheduler
            if init_image is None and mask_image is None:
                image = txt_to_img_engine(
                    prompt=prompt,
                    init_image=None,
                    mask=None,
                    strength=0.5,
                    num_inference_steps=steps,
                    guidance_scale=7.5,
                    eta=0,
                )
            else:
                image = img_to_img_engine(
                    prompt=prompt,
                    init_image=None if init_image is None else cv2.imread(init_image),
                    mask=None if mask_image is None else cv2.imread(mask_image, 0),
                    strength=0.5,
                    num_inference_steps=steps,
                    guidance_scale=7.5,
                    eta=0,
                )

            cv2.imwrite("img.jpg", image)

            with open("img.jpg", "rb") as f:
                content = f.read()
                encoded = base64.b64encode(content).decode("utf-8")
            generated_images.append(encoded)
        prediction["images"] = generated_images
        return prediction
    
    except:
        return {"error":str(traceback.format_exc())}
