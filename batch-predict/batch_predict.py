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

from stable_diffusion_tf.stable_diffusion import StableDiffusion
from PIL import Image
import argparse
def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_height",
        type=int,
        nargs="?",
        required=True,
        help="height of image",
    )
    parser.add_argument(
        "--img_width",
        type=int,
        nargs="?",
        required=True,
        help="width of image",
    )
    parser.add_argument(
        "--jit_compile", 
        type=bool, 
        default=False, 
        help="whether to jit compile or not"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        nargs="?",
        required=True,
        help="number of steps that will execute to produce the image",
    )
    parser.add_argument(
        "--unconditional_guidance_scale",
        type=float,
        nargs="?",
        default=7.5,
        help="hyper-parameter to tune the model",
    )
    parser.add_argument(
        "--temperature",
        type=int,
        nargs="?",
        default=1,
        help="hyper-parameter to tune the model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        default=1,
        help="hyper-parameter to tune the model",
    )
    parser.add_argument(
        "--image_or_text",
        type=str,
        nargs="?",
        default='txt',
        help="whether to generate the image from prompt or another image",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default='An astronaut riding a horse',
        help="the text of what to generate as an image",
    )
    parser.add_argument(
        "--path_to_image",
        type=str,
        nargs="?",
        default='',
        help="the path to the image which will be used as a template for generating the image",
    )
    parser.add_argument(
        "--second_prompt",
        type=str,
        nargs="?",
        default='A Halloween Bedroom',
        help="the prompt to use when making an image from another image",
    )
    args = parser.parse_args()
    return args

def main():
    args = argument_parser()
    img_height = int(args.img_height)
    img_width = int(args.img_width)
    jit_compile = bool(args.jit_compile)
    num_steps = int(args.num_steps)
    u_g_c = float(args.unconditional_guidance_scale)
    temp = float(args.temperature)
    batch_size = int(args.batch_size)
    img_text = args.image_or_text
    prompt = args.prompt.replace('_',' ')
    input_image_path = args.path_to_image
    second_prompt = args.second_prompt.replace('_',' ')
    generator = StableDiffusion(
        img_height=img_height,
        img_width=img_width,
        jit_compile=jit_compile,
    )
    if img_text == 'txt' or img_text == 'text':
        img = generator.generate(
            prompt,
            num_steps=num_steps,
            unconditional_guidance_scale=u_g_c,
            temperature=temp,
            batch_size=batch_size,
        )
    else:
        img = generator.generate(
            second_prompt,
            num_steps=num_steps,
            unconditional_guidance_scale=u_g_c,
            temperature=temp,
            batch_size=batch_size,
            input_image=input_image_path
        )
    Image.fromarray(img[0]).save("/cnvrg/output.png")

if __name__ == "__main__":
    main()
    