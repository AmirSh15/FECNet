import pandas as pd
import urllib
import cv2, os
import requests
from multiprocessing import Pool

def get_img(name_dic, verbose=True):
    id, name_dic = name_dic
    for key, value in name_dic.items():
        if not os.path.isfile("data/train/" + key.split('/')[-1]):
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.62 Safari/537.36'}
                response = requests.get(key, headers=headers)
                if response.status_code:
                    data = response.content
                    check_chars = data[-2:]

                    if check_chars != b'\xff\xd9':
                        if verbose:
                            print('Not complete image, failed to download: ' + str(key))
                        return id, False
                    else:
                        name = "data/train/" + key.split('/')[-1]
                        f = open(name, 'wb')
                        f.write(data)
                        f.close()

                        image = cv2.imread(name)
                        if image is not None:
                            x = image.shape
                            points = name_dic.get(key)
                            image = image[int(points[2] * x[0]):int(points[3] * x[0]),
                                    int(points[0] * x[1]):int(points[1] * x[1])]
                            res = cv2.resize(image, (224, 224))
                            cv2.imwrite(name, res)
                            if verbose:
                                print("Donwloaded image with url:", key)
            except urllib.error.HTTPError as e:
                if e.code == 410:
                    if verbose:
                        print("Hoops")
                if e.code == 403:
                    if verbose:
                        print("Too fast")
                return id, False
    return id, True

def download_img():
    # name_dic1, name_dic2, name_dic3, name_dic4, name_dic5, name_dic6, name_dic7, name_dic8 = load()
    dataset = pd.read_csv('data/FEC_dataset/faceexp-comparison-data-train-public.csv', header=None,
                          error_bad_lines=False)
    
    # number of parallel threads
    steps = 64

    # get_img([0, {dataset.iloc[0, 0]:[dataset.iloc[0, 1], dataset.iloc[0, 2], dataset.iloc[0, 3], dataset.iloc[0, 4]],
    #             dataset.iloc[0, 5]:[dataset.iloc[0, 6], dataset.iloc[0, 7], dataset.iloc[0, 8], dataset.iloc[0, 9]],
    #             dataset.iloc[0, 10]:[dataset.iloc[0, 11], dataset.iloc[0, 12], dataset.iloc[0, 13], dataset.iloc[0, 14]],
    #             }])

    # create list of images to download
    images_add = []
    download_status = [False] * len(dataset)
    for id, data in dataset.iterrows():
        # check if image is already downloaded
        if not os.path.isfile("data/train/" + data[0].split('/')[-1]) or\
                not os.path.isfile("data/train/" + data[5].split('/')[-1]) or\
                not os.path.isfile("data/train/" + data[10].split('/')[-1]):
            images_add.append([[id, {data[0]: [data[1], data[2], data[3], data[4]],
                                data[5]: [data[6], data[7], data[8], data[9]],
                                data[10]: [data[11], data[12], data[13], data[14]],
                                }]])
        else:
            download_status[id] = True
    print("Number of all images:", len(dataset)*3)
    print("Number of images to download:", len(images_add)*3)

    # parallelize the download
    with Pool(steps) as p:
        print("Start parallel downloading")
        outputs = p.starmap(get_img,
                            images_add)

    # update download status
    for id, status in outputs:
        download_status[id] = status

    print("Number of downloaded images:", len(os.listdir("data/train/")))

    # update dataset with download status and save it
    if not os.path.isfile("data/FEC_dataset/faceexp-comparison-data-train-public-downloaded_after_download.csv"):
        dataset[download_status].to_csv(
            "data/FEC_dataset/faceexp-comparison-data-train-public-downloaded_after_download.csv", header=None,
            index=None)
