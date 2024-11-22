import subprocess


def uninstall_tensorflow():
    try:
        # 使用 subprocess.run 执行 pip uninstall 命令
        result = subprocess.run(['pip', 'uninstall', '-y', 'tensorflow'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("卸载成功")
        print("输出:", result.stdout.decode('utf-8'))
    except subprocess.CalledProcessError as e:
        print("卸载失败")
        print("错误:", e.stderr.decode('utf-8'))

# 调用函数
uninstall_tensorflow()
import os

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import pickle
import cv2
import os
import re
import pickle
from moviepy.editor import VideoFileClip, ImageSequenceClip
import subprocess

def set_prompt_and_get_coordinates(output_video, texts=['men', 'the table']):
    if isinstance(texts, str):
        texts = texts.split(',')  # Assuming prompts are separated by commas
        texts = [text.strip() for text in texts]
        print(texts)
    # 保存提示词到文件
    with open('/workspace/texts.pkl', 'wb') as file:
        pickle.dump(texts, file)
    with open('/workspace/output_video.pkl', 'wb') as file:
        pickle.dump(output_video, file)
    # 构建命令
    command = ['python', '/workspace/segment-anything-2-real-time-main/f.py']

    # 执行命令并捕获输出
    all_ok_bboxes = subprocess.run(command, capture_output=True, text=True)
    #print(all_ok_bboxes)
    return all_ok_bboxes

# 示例调用
output_video = '/workspace/segment-anything-2-real-time-main/test-videos/小莫：看我变变变！莫雷高德VS费利克斯_007.mp4'
texts="men, the table"
result = set_prompt_and_get_coordinates(output_video,texts)
print(result)


def run_sam2(output_video):

    # 定义脚本路径
    script_path = '/workspace/segment-anything-2-real-time-main/script-sam2.py'
    # 构建命令
    command = ['python3', script_path ]
    # 执行命令并捕获输出
    sam2_output = subprocess.run(command, capture_output=True, text=True)
    return sam2_output

result = run_sam2(output_video)
print(result)


def run_imges_to_video():

    # 定义脚本路径
    script_path = '/workspace/segment-anything-2-real-time-main/imgtovideos.py'
    # 构建命令
    command = ['python3', script_path ]
    # 执行命令并捕获输出
    sam2_output = subprocess.run(command, capture_output=True, text=True)
    return sam2_output

result = run_imges_to_video()
print(result)
