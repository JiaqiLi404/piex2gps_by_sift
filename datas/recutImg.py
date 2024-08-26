# @Time : 2022/8/18 17:16 
# @Author : Li Jiaqi
# @Description :
import cv2
import os
import Config

for i in os.listdir('data'):
    img = cv2.imread(os.path.join('data', i))
    print(img)
    img=img[250:-250,250:-250]
    cv2.imwrite(os.path.join('data',"recut"+i),img)
