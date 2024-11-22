import os
from moviepy.editor import VideoFileClip, concatenate_audioclips
import os

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import subprocess

def install_tensorflow():
    try:
        # 使用 subprocess.run 执行 pip install 命令
        result = subprocess.run(['pip', 'install', 'tensorflow'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("安装成功")
        print("输出:", result.stdout.decode('utf-8'))
    except subprocess.CalledProcessError as e:
        print("安装失败")
        print("错误:", e.stderr.decode('utf-8'))

# 调用函数
install_tensorflow()
import shutil
import os
import sys
os.chdir("/workspace/transnetv2")
sys.path.append('/workspace/transnetv2')

from moviepy.editor import *
from transnetv2 import TransNetV2
import shutil
import pickle
import json
import pickle


import os
from moviepy.editor import VideoFileClip
import pickle
import time
os.makedirs('/workspace/o_videos', exist_ok=True)
# @title 分割剪辑视频的代码

def check_scenes(video_name):
    scenes_dir = '/workspace/scenes'
    if not os.path.exists(scenes_dir):
        os.makedirs(scenes_dir)
        print("创建目录 %s 成功" % scenes_dir)
    else:
        print("目录 %s 已存在" % scenes_dir)
    if os.path.exists(os.path.join(scenes_dir, video_name)):
        print("文件scenes %s 已存在直接使用" % video_name)
        with open(os.path.join(scenes_dir, video_name), 'rb') as f:
            scenes = pickle.load(f)
        return scenes
    else:
        return False

def change_video_speed(video_path, output_folder):
    video_name = os.path.basename(video_path)
    video_folder = os.path.dirname(video_path)
    scenes = check_scenes(video_name)

    if scenes is not False:
        print("场景检测之前已完成")
    else:
        print("场景检测开始")
        model = TransNetV2()
        video_frames, single_frame_predictions, all_frame_predictions = model.predict_video_2(video_path)
        scenes = model.predictions_to_scenes(single_frame_predictions)
        scenes_dir = '/workspace/scenes'
        with open(os.path.join(scenes_dir, video_name), 'wb') as f:
            pickle.dump(scenes, f)

    print(scenes)
    video_clip2 = VideoFileClip(video_path)
    split_time = scenes
    n = 1
    for i in split_time:
        #time.sleep(1)
        start = i[0]
        end = i[1]
        start_time = start / video_clip2.fps
        end_time = end / video_clip2.fps
        segment_clip = video_clip2.subclip(start_time, end_time)
        output_video_name = f"{os.path.splitext(video_name)[0]}_{n}{os.path.splitext(video_name)[1]}"
        #output_video_name = f"{n}{os.path.splitext(video_name)[1]}"
        output_video_path = os.path.join(output_folder, output_video_name)
        #segment_clip.write_videofile(output_video_path, fps=video_clip2.fps)
        segment_clip.write_videofile(output_video_path, fps=15, codec='libx264', audio_codec='aac')
        n += 1

change_video_speed('/workspace/林诗栋输张本智和，输在了哪儿？.mp4', '/workspace/o_videos')