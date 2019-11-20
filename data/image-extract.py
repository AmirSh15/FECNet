import pandas as pd
from pprint import pprint
import urllib
import numpy as np
import cv2
from threading import Thread

dataset = pd.read_csv('faceexp-comparison-data-train-public.csv',header=None,error_bad_lines=False)
pprint(dataset)
#print(dataset[0])
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.62 Safari/537.36'}
# req = urllib.request.Request(url=str(dataset.iloc[1,0]), headers=headers)  # 必须使用url=url，headers=headers的格式，否则报错
# print(dataset.iloc[1,0])
# response = urllib.request.urlopen(req)
# data = response.read()
# name = dataset.iloc[1,0].split('/')[-1]
# f = open(name,'wb')
# f.write(data)
# f.close()
# image = cv2.imread(name)
# x = image.shape
# image = image[int(dataset.iloc[1,3]*x[0]):int(dataset.iloc[1,4]*x[0]),int(dataset.iloc[1,1]*x[1]):int(dataset.iloc[1,2]*x[1])]
# pprint(image)
# print(image.shape)
# res=cv2.resize(image,(224,224))
# cv2.imwrite(name,res)
# print(res.shape)

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


# for i in range(0,446535):
#     req1 = urllib.request.Request(url=str(dataset.iloc[i, 0]), headers=headers)  # 必须使用url=url，headers=headers的格式，否则报错
#     req2 = urllib.request.Request(url=str(dataset.iloc[i, 5]), headers=headers)
#     req3 = urllib.request.Request(url=str(dataset.iloc[i, 10]), headers=headers)
#     #print(dataset.iloc[i, 0])
#     try:
#         response1 = urllib.request.urlopen(req1)
#         data1 = response1.read()
#         response2 = urllib.request.urlopen(req2)
#         data2 = response2.read()
#         response3 = urllib.request.urlopen(req3)
#         data3 = response3.read()
#
#         name1 = "train/" + dataset.iloc[i, 0].split('/')[-1]
#         name2 = "train/" + dataset.iloc[i, 5].split('/')[-1]
#         name3 = "train/" + dataset.iloc[i, 10].split('/')[-1]
#
#         f1 = open(name1, 'wb')
#         f1.write(data1)
#         f1.close()
#         image1 = cv2.imread(name1)
#         x1 = image1.shape
#         image1 = image1[int(dataset.iloc[i, 3] * x1[0]):int(dataset.iloc[i, 4] * x1[0]),
#                  int(dataset.iloc[i, 1] * x1[1]):int(dataset.iloc[i, 2] * x1[1])]
#         print(image1.shape)
#         res1 = cv2.resize(image1, (224, 224))
#         cv2.imwrite(name1, res1)
#         #print(res1.shape)
#
#         f2 = open(name2, 'wb')
#         f2.write(data2)
#         f2.close()
#         image2 = cv2.imread(name2)
#         x2 = image2.shape
#         image2 = image2[int(dataset.iloc[i, 8] * x2[0]):int(dataset.iloc[i, 9] * x2[0]),
#                  int(dataset.iloc[i, 6] * x2[1]):int(dataset.iloc[i, 7] * x2[1])]
#         print(image2.shape)
#         res2 = cv2.resize(image2, (224, 224))
#         cv2.imwrite(name2, res2)
#         #print(res2.shape)
#
#         f3 = open(name3, 'wb')
#         f3.write(data3)
#         f3.close()
#         image3 = cv2.imread(name3)
#         x3 = image3.shape
#         image3 = image3[int(dataset.iloc[i, 13] * x3[0]):int(dataset.iloc[i, 14] * x3[0]),
#                  int(dataset.iloc[i, 11] * x3[1]):int(dataset.iloc[i, 12] * x3[1])]
#         print(image3.shape)
#         res3 = cv2.resize(image3, (224, 224))
#         cv2.imwrite(name3, res3)
#         #print(res3.shape)
#
#     except urllib.error.HTTPError as e:
#         if e.code == 410:
#             continue

class get_img(Thread):
    def __init__(self,name_dic):
        super().__init__()
        self.name_dic = name_dic
    def run(self):
        for key, value in self.name_dic.items():
            try:
                req = urllib.request.Request(url=str(key), headers=headers)  # 必须使用url=url，headers=headers的格式，否则报错
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
                print(image.shape)
                res = cv2.resize(image, (224, 224))
                cv2.imwrite(name, res)
            except urllib.error.HTTPError as e:
                if e.code == 410:
                    print("Hoops")
                    continue
                if e.code == 403:
                    # time.sleep(5)
                    print("Too fast")
                    continue


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
   print ("Error: 无法启动线程")


