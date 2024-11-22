import os
import subprocess
import shutil
import pickle
import cv2
import re
from moviepy.editor import VideoFileClip, ImageSequenceClip
from tqdm import tqdm

def uninstall_tensorflow():
    try:
        result = subprocess.run(['pip', 'uninstall', '-y', 'tensorflow'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("卸载成功")
        print("输出:", result.stdout.decode('utf-8'))
    except subprocess.CalledProcessError as e:
        print("卸载失败")
        print("错误:", e.stderr.decode('utf-8'))

# 调用函数
uninstall_tensorflow()

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def set_prompt_and_get_coordinates(output_video, texts=['men', 'the table']):
    if isinstance(texts, str):
        texts = texts.split(',')
        texts = [text.strip() for text in texts]
        print(texts)
    with open('/workspace/texts.pkl', 'wb') as file:
        pickle.dump(texts, file)
    with open('/workspace/output_video.pkl', 'wb') as file:
        pickle.dump(output_video, file)
    command = ['python', '/workspace/segment-anything-2-real-time-main/f.py']
    all_ok_bboxes = subprocess.run(command, capture_output=True, text=True)
    return all_ok_bboxes

def run_sam2(output_video):
    script_path = '/workspace/segment-anything-2-real-time-main/script-sam2.py'
    command = ['python3', script_path]
    sam2_output = subprocess.run(command, capture_output=True, text=True)
    return sam2_output

def run_imges_to_video():
    script_path = '/workspace/segment-anything-2-real-time-main/imgtovideos.py'
    command = ['python3', script_path]
    sam2_output = subprocess.run(command, capture_output=True, text=True)
    return sam2_output

def process_videos(source_dir, destination_dir, texts="men, the table"):
    # 确保目标目录存在
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # 获取源目录中的所有视频文件
    video_files = [f for f in os.listdir(source_dir) if f.endswith(('.mp4', '.avi', '.mov'))]

    # 使用 tqdm 包装循环以显示进度条
    for video_file in tqdm(video_files, desc="处理视频", unit="视频"):
        video_path = os.path.join(source_dir, video_file)
        output_video = video_path  # 假设输出视频与输入视频同名

        # 处理视频
        result = set_prompt_and_get_coordinates(output_video, texts)
        print(result.stdout)

        result = run_sam2(output_video)
        print(result)

        result = run_imges_to_video()
        print(result.stdout)

        # 移动原视频到目标目录
        shutil.move(video_path, os.path.join(destination_dir, video_file))

# 示例调用
source_dir = '/workspace/o_videos'
destination_dir = '/workspace/alreayvideos'
process_videos(source_dir, destination_dir)