import pandas as pd
from pprint import pprint
import urllib
import numpy as np
import cv2
from threading import Thread


def load():
    dataset = pd.read_csv('data/FEC_dataset/faceexp-comparison-data-train-public.csv',header=None,error_bad_lines=False)
    # pprint(dataset)

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.62 Safari/537.36'}


    name_dic1 = {}
    name_dic2 = {}
    name_dic3 = {}
    name_dic4 = {}
    name_dic5 = {}
    name_dic6 = {}
    name_dic7 = {}
    name_dic8 = {}
    for i in range(0,55817):
        name1 = dataset.iloc[i, 0]
        name2 = dataset.iloc[i, 5]
        name3 = dataset.iloc[i, 10]
        name_dic1[name1] = [dataset.iloc[i, 1],dataset.iloc[i, 2],dataset.iloc[i, 3],dataset.iloc[i, 4]]
        name_dic1[name2] = [dataset.iloc[i, 6],dataset.iloc[i, 7],dataset.iloc[i, 8],dataset.iloc[i, 9]]
        name_dic1[name3] = [dataset.iloc[i, 11],dataset.iloc[i, 12],dataset.iloc[i, 13],dataset.iloc[i, 14]]

    for i in range(55817,111634):
        name1 = dataset.iloc[i, 0]
        name2 = dataset.iloc[i, 5]
        name3 = dataset.iloc[i, 10]
        name_dic2[name1] = [dataset.iloc[i, 1],dataset.iloc[i, 2],dataset.iloc[i, 3],dataset.iloc[i, 4]]
        name_dic2[name2] = [dataset.iloc[i, 6],dataset.iloc[i, 7],dataset.iloc[i, 8],dataset.iloc[i, 9]]
        name_dic2[name3] = [dataset.iloc[i, 11],dataset.iloc[i, 12],dataset.iloc[i, 13],dataset.iloc[i, 14]]


    for i in range(111634,167451):
        name11 = dataset.iloc[i, 0]
        name12 = dataset.iloc[i, 5]
        name13 = dataset.iloc[i, 10]
        name_dic3[name11] = [dataset.iloc[i, 1],dataset.iloc[i, 2],dataset.iloc[i, 3],dataset.iloc[i, 4]]
        name_dic3[name12] = [dataset.iloc[i, 6],dataset.iloc[i, 7],dataset.iloc[i, 8],dataset.iloc[i, 9]]
        name_dic3[name13] = [dataset.iloc[i, 11],dataset.iloc[i, 12],dataset.iloc[i, 13],dataset.iloc[i, 14]]


    for i in range(167451,223268):
        name11 = dataset.iloc[i, 0]
        name12 = dataset.iloc[i, 5]
        name13 = dataset.iloc[i, 10]
        name_dic4[name11] = [dataset.iloc[i, 1],dataset.iloc[i, 2],dataset.iloc[i, 3],dataset.iloc[i, 4]]
        name_dic4[name12] = [dataset.iloc[i, 6],dataset.iloc[i, 7],dataset.iloc[i, 8],dataset.iloc[i, 9]]
        name_dic4[name13] = [dataset.iloc[i, 11],dataset.iloc[i, 12],dataset.iloc[i, 13],dataset.iloc[i, 14]]


    for i in range(223268,279085):
        name21 = dataset.iloc[i, 0]
        name22 = dataset.iloc[i, 5]
        name23 = dataset.iloc[i, 10]
        name_dic5[name21] = [dataset.iloc[i, 1],dataset.iloc[i, 2],dataset.iloc[i, 3],dataset.iloc[i, 4]]
        name_dic5[name22] = [dataset.iloc[i, 6],dataset.iloc[i, 7],dataset.iloc[i, 8],dataset.iloc[i, 9]]
        name_dic5[name23] = [dataset.iloc[i, 11],dataset.iloc[i, 12],dataset.iloc[i, 13],dataset.iloc[i, 14]]


    for i in range(223268,334901):
        name21 = dataset.iloc[i, 0]
        name22 = dataset.iloc[i, 5]
        name23 = dataset.iloc[i, 10]
        name_dic6[name21] = [dataset.iloc[i, 1],dataset.iloc[i, 2],dataset.iloc[i, 3],dataset.iloc[i, 4]]
        name_dic6[name22] = [dataset.iloc[i, 6],dataset.iloc[i, 7],dataset.iloc[i, 8],dataset.iloc[i, 9]]
        name_dic6[name23] = [dataset.iloc[i, 11],dataset.iloc[i, 12],dataset.iloc[i, 13],dataset.iloc[i, 14]]


    for i in range(334901,390718):
        name31 = dataset.iloc[i, 0]
        name32 = dataset.iloc[i, 5]
        name33 = dataset.iloc[i, 10]
        name_dic7[name31] = [dataset.iloc[i, 1],dataset.iloc[i, 2],dataset.iloc[i, 3],dataset.iloc[i, 4]]
        name_dic7[name32] = [dataset.iloc[i, 6],dataset.iloc[i, 7],dataset.iloc[i, 8],dataset.iloc[i, 9]]
        name_dic7[name33] = [dataset.iloc[i, 11],dataset.iloc[i, 12],dataset.iloc[i, 13],dataset.iloc[i, 14]]

    for i in range(390718,446535):
        name31 = dataset.iloc[i, 0]
        name32 = dataset.iloc[i, 5]
        name33 = dataset.iloc[i, 10]
        name_dic8[name31] = [dataset.iloc[i, 1],dataset.iloc[i, 2],dataset.iloc[i, 3],dataset.iloc[i, 4]]
        name_dic8[name32] = [dataset.iloc[i, 6],dataset.iloc[i, 7],dataset.iloc[i, 8],dataset.iloc[i, 9]]
        name_dic8[name33] = [dataset.iloc[i, 11],dataset.iloc[i, 12],dataset.iloc[i, 13],dataset.iloc[i, 14]]

    return name_dic1, name_dic2, name_dic3, name_dic4, name_dic5, name_dic6, name_dic7, name_dic8

class get_img(Thread):
    def __init__(self,name_dic):
        super().__init__()
        self.name_dic = name_dic
    def run(self):
        for key, value in self.name_dic.items():
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.62 Safari/537.36'}
                req = urllib.request.Request(url=str(key), headers=headers)
                print(key)
                response = urllib.request.urlopen(req)
                data = response.read()
                name = "data/train/" + key.split('/')[-1]
                f = open(name, 'wb')
                f.write(data)
                f.close()
                image = cv2.imread(name)
                x = image.shape
                points = self.name_dic.get(key)
                image = image[int(points[2] * x[0]):int(points[3] * x[0]),
                        int(points[0] * x[1]):int(points[1] * x[1])]
                # pprint(image)
                # print(image.shape)
                res = cv2.resize(image, (224, 224))
                cv2.imwrite(name, res)
            except urllib.error.HTTPError as e:
                if e.code == 410:
                    print("Hoops")
                    continue
                if e.code == 403:
                    print("Too fast")
                    continue

def download_img():
    name_dic1, name_dic2, name_dic3, name_dic4, name_dic5, name_dic6, name_dic7, name_dic8 = load()
    try:
       t1 = get_img(name_dic1)
       t2 = get_img(name_dic2)
       t3 = get_img(name_dic3)
       t4 = get_img(name_dic4)
       t5 = get_img(name_dic5)
       t6 = get_img(name_dic6)
       t7 = get_img(name_dic7)
       t8 = get_img(name_dic8)

       t1.start()
       t2.start()
       t3.start()
       t4.start()
       t5.start()
       t6.start()
       t7.start()
       t8.start()

    except:
       print ("No internet connection")




