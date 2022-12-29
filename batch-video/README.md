# Video Generation using Stable Diffusion
Stable Diffusion is a latent text-to-image diffusion model. Thanks to a generous compute donation from Stability AI and support from LAION, the creators of stable diffusion were able to train a Latent Diffusion Model on 512x512 images from a subset of the LAION-5B database. Similar to Google's Imagen, this model uses a frozen CLIP ViT-L/14 text encoder to condition the model on text prompts. With its 860M UNet and 123M text encoder, the model is relatively lightweight and runs on a GPU with at least 10GB VRAM.

This library is a specific version of stable diffusion optimized on tensorflow [here](https://github.com/divamgupta/stable-diffusion-tensorflow)

### Features
- can generate videos from text (prompts).

# Input Arguments
- `--project_name` name of the project name under which the video and frames will be saved
- `--max_frame_cnt` maximum number of frames that will be used within the video
- `--no_of_prompts_start_frames` number of prompts that will create the video and the starting number of each prompt.
- `--angle` angle at which the successive frames will be rotated at.
- `--zoom` degree of zoom at which the successive frames will be zoomed in.
- `--translation_x` horizontal displacement of the successive frames.
- `--translation_y` vertical displacement of the successive frames.
- `--seed_behavior` iterative or fixed seed increment
- `--seed` starting seed value
- `--fps` frames per second of the video
- `--diffusion_style` which specific diffusion style will be used for the frame generation
- `--prompt_texts` individual prompts in text form that will create the video
- `--music_file_path` path to the file that will contain the music for the frame.

# Model Artifacts
- `--final_video.mp4` the final video that will be outputted after the library finishes running.

# References
a. [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
b. [Stable Diffusion Version Used Here](https://github.com/divamgupta/stable-diffusion-tensorflow)
 