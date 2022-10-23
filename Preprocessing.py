import os
from data.export_train_label import creat_label
# from data.image_downloader import download_img
from data.image_downloader_new import download_img

if __name__ == '__main__':
    if not os.path.exists('data/train'):
        os.makedirs('data/train', exist_ok=True)
    # if data/train is empty, download images
    download_img()
    if not os.path.exists('data/labels.csv'):
        creat_label()