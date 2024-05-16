import os
import requests
from tqdm import tqdm

def download_ffhq(output_dir, num_images=70000):
    """
    下载FFHQ数据集。

    参数:
    output_dir (str): 保存图像的目录。
    num_images (int): 要下载的图像数量，默认是70000。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    base_url = 'https://github.com/NVlabs/ffhq-dataset/raw/master/thumbnails128x128/'

    for i in tqdm(range(num_images), desc="Downloading FFHQ dataset"):
        # 构造图像文件名
        img_filename = f'{i:05d}.png'
        img_url = base_url + img_filename
        img_path = os.path.join(output_dir, img_filename)

        # 下载并保存图像
        response = requests.get(img_url, stream=True)
        if response.status_code == 200:
            with open(img_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        else:
            print(f"Failed to download {img_filename}")

# 使用示例
output_dir = 'ffhq_images'
download_ffhq(output_dir)
