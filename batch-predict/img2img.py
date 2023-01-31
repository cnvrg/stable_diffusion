# Copyright (c) 2023 Intel Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# SPDX-License-Identifier: MIT

import argparse
from stable_diffusion_tf.stable_diffusion import StableDiffusion
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument(
    "--prompt",
    type=str,
    nargs="?",
    required=True,
    help="the prompt to render",
)

parser.add_argument(
    "--negative-prompt",
    type=str,
    help="the negative prompt to use (if any)",
)

parser.add_argument(
    "--steps", 
    type=int, 
    default=50, 
    help="number of ddim sampling steps"
)

parser.add_argument(
    "--input",
    type=str,
    nargs="?",
    required=True,
    help="the input image filename",
)

parser.add_argument(
    "--output",
    type=str,
    nargs="?",
    default="img2img-out.jpeg",
    help="the output image filename",
)

args = parser.parse_args()

generator = StableDiffusion(
    img_height=512,
    img_width=512,
    jit_compile=False,  # You can try True as well (different performance profile)
)

img = generator.generate(
    args.prompt,
    negative_prompt=args.negative_prompt,
    num_steps=args.steps,
    unconditional_guidance_scale=7.5,
    temperature=1,
    batch_size=1,
    input_image=args.input,
    input_image_strength=0.8
)
Image.fromarray(img[0]).save(args.output)
