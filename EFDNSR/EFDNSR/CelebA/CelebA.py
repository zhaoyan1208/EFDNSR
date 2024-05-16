import os
import zipfile
import requests


def download_celeba_dataset(output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define the GitHub release URL for CelebA dataset
    url = 'https://github.com/jacobgil/keras-grad-cam/releases/download/v1.0/celeba-dataset.zip'

    # Path to save the downloaded file
    zip_path = os.path.join(output_dir, 'celeba-dataset.zip')

    # Download the CelebA dataset zip file using requests
    response = requests.get(url)
    with open(zip_path, 'wb') as f:
        f.write(response.content)

    # Extract the downloaded zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        print("Extracting CelebA dataset...")
        zip_ref.extractall(output_dir)
        print("Extraction completed.")

    # Remove the zip file after extraction
    os.remove(zip_path)
    print("Downloaded CelebA dataset is located in:", output_dir)


# Specify the output directory to save the CelebA dataset
output_directory = ''

# Call the function to download the CelebA dataset
download_celeba_dataset(output_directory)
