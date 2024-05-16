import os
from sklearn.datasets import fetch_lfw_people

def download_lfw_data(data_home='lfw_data', min_faces_per_person=70, resize=0.4):
    """
    下载LFW数据集并保存到指定目录。

    参数：
    data_home (str): 保存数据集的目录。
    min_faces_per_person (int): 包含最少数量面孔的人的图像。
    resize (float): 图像的缩放因子。
    """
    # 确保数据目录存在
    if not os.path.exists(data_home):
        os.makedirs(data_home)

    # 下载LFW数据集
    lfw_people = fetch_lfw_people(data_home=data_home, min_faces_per_person=min_faces_per_person, resize=resize)

    # 获取图像和元数据
    images = lfw_people.images
    target_names = lfw_people.target_names
    targets = lfw_people.target

    # 保存图像和元数据
    for i, image in enumerate(images):
        person_folder = os.path.join(data_home, target_names[targets[i]])
        if not os.path.exists(person_folder):
            os.makedirs(person_folder)

        image_filename = os.path.join(person_folder, f'image_{i:04d}.jpg')
        Image.fromarray(image).save(image_filename)
        print(f'Saved {image_filename}')

    print('LFW数据集下载并保存完成')

if __name__ == '__main__':
    download_lfw_data()
