import os
import requests
from tqdm import tqdm
from zipfile import ZipFile

def download_file(url, dest_folder, filename=None):
    """
    从指定的URL下载文件并保存到目标文件夹。

    参数：
    url (str): 文件的URL。
    dest_folder (str): 目标文件夹路径。
    filename (str): 保存的文件名，如果未提供，则从URL中推断。
    """
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    if filename is None:
        filename = os.path.basename(url)

    file_path = os.path.join(dest_folder, filename)

    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    with open(file_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR: Something went wrong")

    return file_path

def extract_zip(file_path, extract_to):
    """
    解压缩指定的ZIP文件到目标文件夹。

    参数：
    file_path (str): ZIP文件路径。
    extract_to (str): 解压缩目标文件夹路径。
    """
    with ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def download_and_extract_div2k(dest_folder='DIV2K'):
    """
    下载并解压缩DIV2K数据集。

    参数：
    dest_folder (str): 保存数据集的目标文件夹路径。
    """
    urls = {
        'train_HR': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip',
        'train_LR_bicubic_X2': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip',
        'train_LR_bicubic_X3': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X3.zip',
        'train_LR_bicubic_X4': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip',
        'valid_HR': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip',
        'valid_LR_bicubic_X2': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip',
        'valid_LR_bicubic_X3': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X3.zip',
        'valid_LR_bicubic_X4': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip'
    }

    for key, url in urls.items():
        print(f"Downloading {key} from {url}")
        zip_file_path = download_file(url, dest_folder, f"{key}.zip")
        print(f"Extracting {zip_file_path}")
        extract_zip(zip_file_path, os.path.join(dest_folder, key))
        os.remove(zip_file_path)
        print(f"Finished processing {key}")

if __name__ == '__main__':
    download_and_extract_div2k()
