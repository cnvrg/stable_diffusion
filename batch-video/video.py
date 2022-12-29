from IPython import display
from stable_diffusion_tf.stable_diffusion import StableDiffusion
from stable_diffusion_tf.video_utils import *
import os
from PIL import Image
import random
from base64 import b64encode
from tensorflow import keras
import subprocess
import argparse
import ffmpeg

def miscellaneous_activities():
    sub_p_res = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(sub_p_res)
    keras.mixed_precision.set_global_policy("mixed_float16")

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_name",
        type=str,
        nargs="?",
        required=True,
        help="name of the project",
    )
    parser.add_argument(
        "--maximum_number_of_frames",
        type=int,
        nargs="?",
        required=True,
        help="maximum number of frames in the video",
    )
    parser.add_argument(
        "--no_of_prompts_start_frames",
        type=str,
        nargs="?",
        required=True,
        help="number of prompts that will fill the video[1,2,3,4] and the frame no where they will start from",
    )
    parser.add_argument(
        "--angle",
        type=str,
        nargs="?",
        required=True,
        help="angle",
    )
    parser.add_argument(
        "--zoom",
        type=str,
        nargs="?",
        required=True,
        help="zoom",
    )
    parser.add_argument(
        "--translation_x",
        type=str,
        nargs="?",
        required=True,
        help="translation_x",
    )
    parser.add_argument(
        "--translation_y",
        type=str,
        nargs="?",
        required=True,
        help="translation_y",
    )
    parser.add_argument(
        "--seed_behavior",
        type=str,
        nargs="?",
        required=True,
        help="seed behavior (iter,fix) comma separated",
    )
    parser.add_argument(
        "--seed",
        type=str,
        nargs="?",
        required=True,
        help="seed value",
    )
    parser.add_argument(
        "--fps",
        type=int,
        nargs="?",
        required=True,
        help="frames per second",
    )
    parser.add_argument(
        "--diffusion_style",
        type=str,
        nargs="?",
        required=True,
        help="which kind of diffusion you would like Default,Illustration-Diffusion, Comic-Diffusion,Superhero-Diffusion",
    )
    parser.add_argument(
        "--prompt_texts",
        type=str,
        nargs="?",
        required=True,
        help="list of prompts you want to give",
    )
    parser.add_argument(
        "--music_file_path",
        type=str,
        nargs="?",
        required=True,
        help="background music of the video",
    )
    args = parser.parse_args()
    return args

def main():
    args = argument_parser()
    project_name = args.project_name
    maximum_number_of_frames = args.maximum_number_of_frames
    number_of_prompts = args.no_of_prompts_start_frames.split('!')[0]
    prompt_frame_indexes = args.no_of_prompts_start_frames.split('!')[1]
    angle = float(args.angle)
    zoom = float(args.zoom)
    translation_x = args.translation_x
    print('step1 '+translation_x)
    translation_x = translation_x.replace('sin','sin(') + ')'
    print('step2 '+translation_x)    
    translation_y = int(args.translation_y)
    prompt_texts = args.prompt_texts
    seed = int(args.seed)
    seed_behavior = args.seed_behavior
    fps = int(args.fps)
    music_file_path = args.music_file_path
    video_length = int(maximum_number_of_frames) // int(fps)
    print('Video Length is'+str(video_length))
    if seed == "-1":
        sd = random.randint(0, 2**32 - 1)
    else:
        sd = int(seed)
    translation_x = generate_frames_translation(translation_x, args.maximum_number_of_frames)
    translation_y = generate_frames_translation(translation_y, args.maximum_number_of_frames)
    style_model = args.diffusion_style
    args = {
    'project_name': project_name,
    'maximum_number_of_frames': int(maximum_number_of_frames),
    'number_of_prompts': number_of_prompts,
    'angle': angle,
    'zoom': zoom,
    'translation_x': translation_x,
    'translation_y': translation_y,
    'seed_behavior': seed_behavior,
    'video_length': video_length,
    'seed': sd,
    'fps': fps
    }
    generator = StableDiffusion( 
    img_height=512,
    img_width=512,
    jit_compile=True,  # You can try False as well (different performance profile)
    )
    prompts_frame_dict = prompts_initialization(number_of_prompts, prompt_frame_indexes, maximum_number_of_frames, create_prompts_frames_dict, prompt_texts)
    frames_creation(project_name, prompts_frame_dict, args, generator)
    video_creation(construct_ffmpeg_video_cmd,args)
    sound_addition(music_file_path,args)

def prompts_initialization(number_of_prompts, prompt_frame_indexes, maximum_number_of_frames, create_prompts_frames_dict, prompt_texts):
    print('prompts_initialization_started')    
    prompt_name = {
        'first_prompt': None,
        'second_prompt': None,
        'third_prompt': None,
        'fourth_prompt': None
    }
    prompt_frame = {
        'first_p_frame': None,
        'second_p_frame': None,
        'third_p_frame': None,
        'fourth_p_frame': None
    }
    cnt=0
    for i in range(int(number_of_prompts)):
        prompt_name[list(prompt_name.keys())[i]] = prompt_texts.split('!')[i].replace('_',' ')
        if prompt_frame_indexes != 'Proportional':
            prompt_frame[list(prompt_frame.keys())[i]] = prompt_frame_indexes.split(',')[i]
        else:
            prompt_frame[list(prompt_frame.keys())[i]] = cnt
            cnt = cnt+maximum_number_of_frames/number_of_prompts

    prompts_frames_dict = create_prompts_frames_dict( 
    prompt_name[list(prompt_name.keys())[0]],
    prompt_frame[list(prompt_frame.keys())[0]],
    prompt_name[list(prompt_name.keys())[1]],
    prompt_frame[list(prompt_frame.keys())[1]],
    prompt_name[list(prompt_name.keys())[2]],
    prompt_frame[list(prompt_frame.keys())[2]],
    prompt_name[list(prompt_name.keys())[3]],
    prompt_frame[list(prompt_frame.keys())[3]])
    return prompts_frames_dict

def video_creation(construct_ffmpeg_video_cmd,args):
    print('video_creation_started')
    image_path = os.path.join(f"/cnvrg/{args['project_name']}", "img_%05d.png")
    mp4_path = os.path.join('/cnvrg', f"{args['project_name']}.mp4")
    construct_ffmpeg_video_cmd(args, image_path, mp4_path)

def sound_addition(music_file_path,args):
    print('sound_addition_started')
    sound_path = music_file_path
    mp3_path = '/cnvrg/music_file.mp3'
    create_audio(args, sound_path, mp3_path)
    vid_path = os.path.join('/cnvrg', args['project_name']+".mp4")
    aud_path = '/cnvrg/music_file.mp3'
    combined_path = '/cnvrg/final_video.mp4'
    construct_ffmpeg_combined_cmd(vid_path, aud_path, combined_path)
    
def frames_creation(project_name, prompts_frames_dict, args, generator):
    print('frames_creation_started')    
    path = os.path.join('/cnvrg',project_name)
    if not os.path.exists(path):
        os.makedirs(path)

    prompt_iprompt_seq_lst = create_prompt_iprompt_seq(args, prompts_frames_dict)
    curr_frame = None
    prev = None

    for prompt_seq in prompt_iprompt_seq_lst:
        prompt_seq_keys_lst = list(prompt_seq.keys())
        prompt = prompt_seq[prompt_seq_keys_lst[0]]
        for i in range(prompt_seq_keys_lst[0], prompt_seq_keys_lst[-1]+1):
            if curr_frame is None:
                curr_frame = generate_init_frame(prompt, args, generator)
                Image.fromarray(curr_frame).save(f"/cnvrg/{args['project_name']}/img_{i:05}.png", format="png")
                curr_frame = anim_frame_warp_2d(curr_frame, args, i)

            if prev is not None:
                curr_frame = maintain_colors(curr_frame, prev)

            prev = curr_frame
            print(prompt)
            img = generator.generate(
                prompt,
                seed=args['seed'],
                num_steps=50,
                unconditional_guidance_scale=7.5,
                temperature=1,
                batch_size=1,
                input_image=prev,
                input_image_strength=0.6,
            )
            Image.fromarray(img[0]).save(f"/cnvrg/{args['project_name']}/img_{i:05}.png", format="png")

            if i % 32 == 0 and i != 0:
                display.clear_output(wait=True)
            display.display(Image.fromarray(img[0]))
            curr_frame = anim_frame_warp_2d(img[0], args, i)
            next_seed(args)
            


if __name__ == "__main__":
    main()
    