# Image Generation using Stable Diffusion
Stable Diffusion is a latent text-to-image diffusion model. Thanks to a generous compute donation from Stability AI and support from LAION, the creators of stable diffusion were able to train a Latent Diffusion Model on 512x512 images from a subset of the LAION-5B database. Similar to Google's Imagen, this model uses a frozen CLIP ViT-L/14 text encoder to condition the model on text prompts. With its 860M UNet and 123M text encoder, the model is relatively lightweight and runs on a GPU with at least 10GB VRAM.

This library is a specific version of stable diffusion optimized on tensorflow [here](https://github.com/divamgupta/stable-diffusion-tensorflow)

### Features
- can generate images from text (prompts).
- can generate images from other images as well

# Input Arguments
- `--img_height` height of the image that will be created
- `--img_width` width of the image that will be created
- `--jit_compile` whether to jit compile or not. Improves the quality of the image
- `--num_steps` number of iterations that will execute to produce the image.
- `--unconditional_guidance_scale` hyper-parameter to tune the model.
- `--temperature` hyper-parameter to tune the model.
- `--batch_size` how many steps will be taken at once.
- `--image_or_text` whether you want to generate image from prompt or another image.
- `--prompt` the text of what to generate as an image.
- `--path_to_image` the path to the image which will be used as a template for generating the image.
- `--second_prompt` the prompt to use when making an image from another imagee.

# Model Artifacts
- `--output.png` the final image that has been generated

# References
a. [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
b. [Stable Diffusion Version Used Here](https://github.com/divamgupta/stable-diffusion-tensorflow)
 