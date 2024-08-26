import os
import csv
import cv2
import numpy as np

csv_has_title = True
# img_code - img_path
image_path_dict = dict()
# img_code - [img_lon,img_lat,
image_info_dict = dict()
a = 2.6 * 1e-3
f = 35


class FlightInfo:
    # pitch附仰角，roll翻滚角，yaw航向角
    def __init__(self, lon, lat, alt, roll, pitch, yaw):
        self.lon = lon
        self.lat = lat
        self.alt = alt
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw


def load_gps_from_csv():
    global image_info_dict, csv_has_title
    files = os.listdir('data')
    file = None
    for f in files:
        if f.endswith('.csv'):
            file = f
    if file is None:
        raise RuntimeError("!!!!!Image doesn't have GPS attributes!!!!!")
    # 读取csv文件
    with open(os.path.join('data', file)) as f:
        csv_f = csv.reader(f)
        image_info_dict = {}
        for row in csv_f:
            if csv_has_title:
                csv_has_title = False
                continue
            # 提取文件名里的数字
            num_filter = filter(str.isdigit, row[0])
            num_list = list(num_filter)
            num_str = "".join(num_list)
            num_int = int(num_str)
            image_info_dict[num_int] = FlightInfo(float(row[2]), float(row[3]), float(row[4]), float(row[5]),
                                                  float(row[6]), float(row[7]))


def load_image_code_and_path():
    global image_path_dict
    files = os.listdir('data')
    for f in files:
        if not f.endswith('.csv'):
            # 提取文件名里的数字
            num_filter = filter(str.isdigit, f)
            num_list = list(num_filter)
            num_str = "".join(num_list)
            num_int = int(num_str)
            image_path_dict[num_int] = os.path.join('data', f)


# 根据航行参数将图像转向正北方
def rotate_img_to_north(img, yaw, w, h, x, y):
    mat = cv2.getRotationMatrix2D((w / 2, h / 2), -yaw, 1)
    img = cv2.warpAffine(img, mat, (w, h))
    # 所求的点也需要旋转
    mat = list(mat)
    mat.append([0, 0, 1])
    mat = np.array(mat)
    coord_target = np.array([[x], [y], [1]])
    coord_target = np.matmul(mat, coord_target)
    # coord_center=np.array([[w/2], [h/2], [1]])
    # coord_center = np.matmul(mat, coord_center)
    return img, list(coord_target)[0], list(coord_target)[1]


# 根据航行参数调整图像中心经纬度
def get_center_gps(info):
    lon, lat, alt = info.lon, info.lat, info.alt
    roll, pitch, yaw = info.roll / 360 * 2 * np.pi, info.pitch / 360 * 2 * np.pi, info.yaw / 360 * 2 * np.pi
    lon_dis = alt * np.tan(pitch) * np.sin(yaw) - alt * np.tan(roll) * np.cos(yaw)
    lat_dis = alt * np.tan(pitch) * np.cos(yaw) + alt * np.tan(roll) * np.sin(yaw)
    delta_lon, delta_lat = distance_to_gps(lon_dis, lat_dis, lat)
    lon = lon + delta_lon
    lat = lat + delta_lat
    return lon, lat


# 将距离转为经纬度
def distance_to_gps(dis_lon, dis_lat, lat_ref):
    lat = dis_lat / 111000
    lon = dis_lon / (111000 * np.cos(lat_ref / 360 * 2 * np.pi))
    return lon, lat


def get_gps(img_code, w, h):
    # 读取文件
    img_ori = cv2.imread(image_path_dict[img_code])
    img_info = image_info_dict[img_code]
    img_w, img_h = img_ori.shape[1], img_ori.shape[0]
    # 根据航线角度，使图片朝向北方
    img_north, w, h = rotate_img_to_north(img_ori, img_info.yaw, img_w, img_h, w, h)
    img_show = cv2.resize(img_north, (800, 500))
    cv2.imshow('ori-' + str(img_code), img_show)
    # 根据三角度，得到图片中心点的经纬度
    # print("bef gps:",img_info.lon,img_info.lat)
    lon, lat = get_center_gps(img_info)
    # print("aft gps:", lon, lat)
    # 根据相机参数计算比例k
    k = a / f * img_info.alt
    lon_distance = k * (w - img_w / 2)
    lat_distance = k * (h - img_h / 2)
    lon_delta, lat_delta = distance_to_gps(lon_distance, lat_distance, img_info.lat)
    lon = lon + lon_delta
    lat = lat - lat_delta
    print(lon, lat)
    return lon, lat


load_gps_from_csv()
load_image_code_and_path()

# get_gps(15, 6032, 811)
# get_gps(16, 5881, 2149)
# get_gps(17, 5411, 3848)
p11 = get_gps(101, 3349, 1848)
p12 = get_gps(102, 2796, 3013)
p13 = get_gps(103, 2586, 4036)
print('----------------')
p21 = get_gps(101, 5424, 1926)
p22 = get_gps(102, 4855, 3190)
p23 = get_gps(103, 4672, 3913)
print('----------------')
p31 = get_gps(101, 6600, 3055)
p32 = get_gps(102, 5955, 4359)
p33 = get_gps(103, 5905, 4922)
# cv2.waitKey(0)

delta_point1=abs(sum(p11) - sum(p12))+abs(sum(p11) - sum(p13))+abs(sum(p13) - sum(p12))
delta_point2=abs(sum(p21) - sum(p22))+abs(sum(p21) - sum(p23))+abs(sum(p22) - sum(p23))
delta_point3=abs(sum(p31) - sum(p32))+abs(sum(p31) - sum(p33))+abs(sum(p32) - sum(p33))
print(delta_point1/3*111000/1.4,delta_point2/3*111000/1.4,delta_point3/3*111000/1.4)
