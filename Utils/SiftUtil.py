import csv
import time
import multiprocessing as mp
import cv2.cv2 as cv2
import numpy as np
from PIL import Image
import exifread
import math
import fractions
import os
import matplotlib.pyplot as plt
from functools import partial

GPSCsv = None


class SiftImageOperator:
    def __init__(self, imgLeft, imgRight, leftGPS, rightGPS, showImg=False,
                 nfeatures=None,
                 nOctaveLayers=None,
                 contrastThreshold=None,
                 edgeThreshold=None,
                 sigma=None):
        """
        :param imgLeft: 主图
        :param imgRight: 辅助图
        :param leftGPS: 主图GPS
        :param rightGPS: 辅助图GPS
        :param showImg:是否展示过程
        :param nfeatures: 特征点数目（算法对检测出的特征点排名，返回最好的nfeatures个特征点）
        :param nOctaveLayers: nOctaveLayers：金字塔中每组的层数（算法中会自己计算这个值）
        :param contrastThreshold: contrastThreshold：过滤掉较差的特征点的对阈值. contrastThreshold越大，返回的特征点越少.
        :param edgeThreshold: 过滤掉边缘效应的阈值. edgeThreshold越大，特征点越多（被过滤掉的越少）.
        :param sigma: 金字塔第0层图像高斯滤波系数.

        opencv默认参数：
        nOctaveLayers =3
        contrastThreshold = 0.04
        edgeThreshold = 10
        sigma =1.6
        """
        self.src_result_pts = None
        self.dst_result_pts = None
        self.piex_distance_h = None
        self.piex_distance_w = None
        self.zoom = 1e8
        self.piex_k = None
        self.piex_sina = None
        self.piex_cosa = None
        self.piex_h = None
        self.piex_w = None
        self.__H = None
        self.imgFinal = None
        self.showImg = showImg
        self.__imgLeft = cv2.cvtColor(np.array(imgLeft), cv2.COLOR_RGB2BGR)
        self.__imgRight = cv2.cvtColor(np.array(imgRight), cv2.COLOR_RGB2BGR)
        self.leftGPS = leftGPS
        self.rightGPS = rightGPS
        self.__grayLeft = cv2.cvtColor(self.__imgLeft, cv2.COLOR_BGR2GRAY)
        self.__grayRight = cv2.cvtColor(self.__imgRight, cv2.COLOR_BGR2GRAY)
        self.__sift = cv2.SIFT_create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma)
        self.leftMedia = [self.__imgLeft.shape[1] // 2, self.__imgLeft.shape[0] // 2]  # w h
        self.coef_w = None
        self.coef_h = None
        self.right_img_k = None
        self.right_img_cosa = None
        self.right_img_sina = None

    @classmethod
    def gen_from_params(cls, imgLeft, imgRight, leftGPSLon, leftGPSLat, rightGPSLon, rightGPSLat, coef_w_0, coef_w_1,
                        coef_h_0, coef_h_1, piex_w, piex_h, piex_k, piex_cosa):
        operator = SiftImageOperator(imgLeft, imgRight, [leftGPSLon, leftGPSLat], [rightGPSLon, rightGPSLat], False)
        operator.piex_cosa = piex_cosa
        operator.piex_sina = (1 - piex_cosa ** 2) ** 0.5
        operator.coef_w = np.array([coef_w_0, coef_w_1])
        operator.coef_h = np.array([coef_h_0, coef_h_1])
        operator.piex_h = piex_h
        operator.piex_w = piex_w
        operator.piex_k = piex_k
        return operator

    @classmethod
    def gen_from_dict(cls, imgLeft, imgRight, dict):
        return cls.gen_from_params(imgLeft, imgRight, float(dict[b"left_lon"]), float(dict[b"left_lat"]),
                                   float(dict[b"right_lon"]), float(dict[b"right_lat"]), float(dict[b"coef_w_0"]),
                                   float(dict[b"coef_w_1"]), float(dict[b"coef_h_0"]), float(dict[b"coef_h_1"]),
                                   float(dict[b"piex_w"]), float(dict[b"piex_h"]), float(dict[b"piex_k"]),
                                   float(dict[b"piex_cosa"]))

    def to_dict(self):
        """
        将类转为dict，方便redis存储
        :return: dict
        """
        return {"coef_w_0": float(self.coef_w[0]), "coef_w_1": float(self.coef_w[1]), "coef_h_0": float(self.coef_h[0]),
                "coef_h_1": float(self.coef_h[1]), "piex_w": self.piex_w, "piex_h": self.piex_h, "piex_k": self.piex_k,
                "piex_cosa": self.piex_cosa, "left_lon": self.leftGPS[0], "left_lat": self.leftGPS[1],
                "right_lon": self.rightGPS[0], "right_lat": self.rightGPS[1]}

    def _siftCompute(self, grayImg):
        img_copy = grayImg.copy()
        keyPoints, describes = self.__sift.detectAndCompute(img_copy, None)
        return keyPoints, describes

    def _KDTreeMatch(self, describesLeft, describesRight):
        # K-D tree建立索引方式的常量参数
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # checks指定索引树要被遍历的次数
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches_1 = flann.knnMatch(describesLeft, describesRight, k=2)  # 进行匹配搜索，参数k为返回的匹配点对数量
        # 把保留的匹配点放入good列表
        good1 = []
        T = 0.5  # 阈值
        # 筛选特征点
        for i, (m, n) in enumerate(matches_1):
            if m.distance < T * n.distance:  # 如果最近邻点的距离和次近邻点的距离比值小于阈值，则保留最近邻点
                good1.append(m)
            #  双向交叉检查方法
        matches_2 = flann.knnMatch(describesRight, describesLeft, k=2)  # 进行匹配搜索
        # 把保留的匹配点放入good2列表
        good2 = []
        for (m, n) in matches_2:
            if m.distance < T * n.distance:  # 如果最近邻点的距离和次近邻点的距离比值小于阈值，则保留最近邻点
                good2.append(m)
        match_features = []  # 存放最终的匹配点
        for i in good1:
            for j in good2:
                if (i.trainIdx == j.queryIdx) & (i.queryIdx == j.trainIdx):
                    match_features.append(i)
        return match_features

    def _getHomography(self, dst_pts, src_pts):
        # 获取视角变换矩阵
        """
         findHomography: 计算多个二维点对之间的最优单映射变换矩阵 H（3行x3列） ，使用最小均方误差或者RANSAC方法
         参考网址： https://blog.csdn.net/fengyeer20120/article/details/87798638
        """
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 10)  # mask中返回的匹配点是否满足最优单映射变换矩阵
        return H, mask

    def _drawMatches(self, imageA, imageB, src_result_pts, dst_result_pts):
        # 初始化可视化图片，将A、B图左右连接到一起
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # 联合遍历，画出匹配对
        for (p1, p2) in zip(src_result_pts, dst_result_pts):
            # 当点对匹配成功时，画到可视化图上
            p2[0][0] = p2[0][0] + wA
            cv2.line(vis, (int(p1[0][0]), int(p1[0][1])), (int(p2[0][0]), int(p2[0][1])), (0, 0, 255), 3)

        # 返回可视化结果
        return vis

    def _getFinalImg(self, H):
        if self.imgFinal is not None:
            return self.imgFinal
        # 得到右图的坐标点
        h1, w1, p1 = self.__imgRight.shape
        h2, w2, p2 = self.__imgLeft.shape

        # 计算四个坐标点
        corner = np.zeros((4, 2))  # 存放四个角坐标，依次为左上角，左下角， 右上角，右下角
        row, col, c = h1, w1, p1

        # 左上角(0, 0, 1)
        v2 = np.array([0, 0, 1])
        v1 = np.dot(H, v2)
        corner[0, 0] = v1[0] / v1[2]
        corner[0, 1] = v1[1] / v1[2]

        # 左下角
        v2[0] = 0
        v2[1] = row
        v1 = np.dot(H, v2)
        corner[1, 0] = v1[0] / v1[2]
        corner[1, 1] = v1[1] / v1[2]

        # 右上角
        v2[0] = col
        v2[1] = 0
        v1 = np.dot(H, v2)
        corner[2, 0] = v1[0] / v1[2]
        corner[2, 1] = v1[1] / v1[2]

        # 右下角
        v2[0] = col
        v2[1] = row
        v1 = np.dot(H, v2)
        corner[3, 0] = v1[0] / v1[2]
        corner[3, 1] = v1[1] / v1[2]

        right_top_x = np.int32(corner[2, 0])
        right_bottom_x = np.int32(corner[3, 0])
        left_top_x = np.int32(corner[0, 0])
        left_bottom_x = np.int32(corner[1, 0])

        right_top_y = np.int32(corner[2, 1])
        right_bottom_y = np.int32(corner[3, 1])
        left_top_y = np.int32(corner[0, 1])
        left_bottom_y = np.int32(corner[1, 1])

        w_max = np.maximum(right_top_x, right_bottom_x)
        w_min = np.minimum(left_bottom_x, left_top_x)
        h_max = np.maximum(left_bottom_y, right_bottom_y)
        h_min = np.minimum(right_top_y, left_top_y)

        print("原图坐标:(0,0),(", h2, ",", w2, "),矫正图坐标:(", left_top_y, ",", left_top_x, "),(", right_bottom_y, ",",
              right_bottom_x, ")")

        # 坐标转换
        if h_min < 0:
            # 补上图像
            imgRightTemp = np.zeros((h2 + np.abs(h_min), w2, p1), dtype=np.uint8)
            imgRightTemp[np.abs(h_min):, :] = self.__imgRight
            if (h1 - h_min) > (h_max - h_min):
                h = h1 - h_min
            else:
                h = h_max - h_min
            self.imgFinal = cv2.warpPerspective(imgRightTemp, H, (w_max, h))  # 坐标转换

            if self.showImg:
                showimg = cv2.resize(self.imgFinal, (1800, 900))
                cv2.imshow("imgright_h<0", showimg)

            self.imgFinal[np.abs(h_min):(np.abs(h_min) + h2), :w2] = self.__imgLeft  # 参考图像高度补齐

        else:
            if self.__imgLeft.shape[0] > h_max:
                h = self.__imgLeft.shape[0]
            else:
                h = h_max
            self.imgFinal = cv2.warpPerspective(self.__imgRight, H, (w_max, h))  # 坐标转换

            if self.showImg:
                showimg = cv2.resize(self.imgFinal, (1800, 900))
                cv2.imshow("imgright_h>0", showimg)

            self.imgFinal[:h2, :w2] = self.__imgLeft

        return self.imgFinal

    def __calculate_angle_k_of_vactor(self, toVc, fromVac):
        def get_angle(vac):
            if vac[0] == 0:
                if vac[1] >= 0:
                    return np.pi / 2
                else:
                    return -np.pi / 2
            if vac[1] >= 0 and vac[0] >= 0:
                # 第一象限
                return np.arctan(vac[1] / vac[0])
            elif vac[1] >= 0 and vac[0] <= 0:
                # 第二象限
                return np.pi - np.arctan(abs(vac[1] / vac[0]))
            elif vac[0] <= 0:
                # 第三象限
                return np.pi + np.arctan(abs(vac[1] / vac[0]))
            else:
                # 第四象限
                return 2 * np.pi - np.arctan(abs(vac[1] / vac[0]))

        [tx, ty] = toVc
        [fx, fy] = fromVac
        # 计算向量比例系数
        piex_distance = (fx ** 2 + fy ** 2) ** 0.5
        gps_distance = (ty ** 2 + tx ** 2) ** 0.5
        piex_k = gps_distance / piex_distance

        # 计算向量角度cosa=gps*piex/|gps|*|piex|  ->  gps=cosa*|gps|*|piex|/piex
        cosa = (tx * fx + ty * fy) / (piex_distance * gps_distance)
        if cosa > 1:
            cosa = 1
        sina = (1 - cosa ** 2) ** 0.5

        if 0 < get_angle(fromVac) - get_angle(toVc) < np.pi or get_angle(fromVac) - get_angle(toVc) < -np.pi:
            sina = -sina

        # print('c', fromVac, toVc,get_angle(fromVac),get_angle(toVc), piex_k, sina, cosa)

        return piex_k, sina, cosa

    def __rotate_vac_to_angle(self, fromVac, sina, cosa, k):
        gps_coord_temp = [k * fromVac[0], k * fromVac[1]]
        # 向量旋转公式：https://blog.csdn.net/zhinanpolang/article/details/82912325
        # x1 = x0 * cosB - y0 * sinB y1 = x0 * sinB + y0 * cosB
        toVac = [gps_coord_temp[0] * cosa - gps_coord_temp[1] * sina,
                 gps_coord_temp[0] * sina + gps_coord_temp[1] * cosa]
        return toVac

    def computeImages(self):
        """
        计算相关点，为gps像素转换做准备，得到两图对应特征点像素值集
        """
        print("-----sift: 开始特征点计算")
        keyPointsLeft, describesLeft = self._siftCompute(self.__grayLeft)
        keyPointsRight, describesRight = self._siftCompute(self.__grayRight)
        print("-----sift: 左图", len(keyPointsLeft), "个特征点，右图", len(keyPointsRight), "个特征点")
        # 特征匹配
        print("-----sift: 开始特征点匹配和视角变换矩阵计算")
        match_features = self._KDTreeMatch(describesLeft, describesRight)

        src_pts = np.float32([keyPointsLeft[m.queryIdx].pt for m in match_features]).reshape(-1, 1, 2)  # 转换成列表
        dst_pts = np.float32([keyPointsRight[m.trainIdx].pt for m in match_features]).reshape(-1, 1, 2)
        # 获取视角变换矩阵
        self.__H, mask = self._getHomography(dst_pts, src_pts)

        # 存放精匹配后的特征点
        self.src_result_pts = []
        self.dst_result_pts = []
        self.piex_distance_w = []
        self.piex_distance_h = []

        # 存放最左右两个特征点的坐标
        src_w = []
        src_h = []
        dst_w = []
        dst_h = []

        for i, value in enumerate(mask):
            if value == 1:
                self.src_result_pts.append(src_pts[i])
                self.dst_result_pts.append(dst_pts[i])
                src_w.append(src_pts[i][0][0])
                src_h.append(src_pts[i][0][1])
                dst_w.append(dst_pts[i][0][0])
                dst_h.append(dst_pts[i][0][1])

        #  将右图旋转到和左图同一朝向
        self.right_img_k = self.right_img_sina = self.right_img_cosa = 0
        for t in range(50):
            x1 = np.random.randint(0, len(dst_w))
            x2 = 0
            while x2 == 0 or x2 == x1:
                x2 = np.random.randint(1, len(dst_w))
            rightVac = [dst_w[x1] - dst_w[x2], dst_h[x1] - dst_h[x2]]
            leftVac = [src_w[x1] - src_w[x2], src_h[x1] - src_h[x2]]
            k, sina, cosa = self.__calculate_angle_k_of_vactor(leftVac, rightVac)
            self.right_img_k += 0.02 * k
            self.right_img_sina += 0.02 * sina
            self.right_img_cosa += 0.02 * cosa

        # height, width = self.__imgRight.shape[:2]
        # H = cv2.getRotationMatrix2D((width/2, height/2), np.arccos(self.right_img_cosa)*180/np.pi, 1)
        # img=cv2.warpAffine(self.__imgRight, H, (width, height))
        # img=cv2.resize(img,(1800,900))
        # cv2.imshow('aft_r', img)
        # cv2.waitKey(0)

        # 中心坐标矫正
        height, width = self.__imgRight.shape[:2]
        mid = [width / 2, height / 2]
        aftmid = self.__rotate_vac_to_angle(mid, self.right_img_sina, self.right_img_cosa, self.right_img_k)
        delmid = [mid[x] - aftmid[x] for x in range(2)]

        # 将右图坐标映射到左图坐标系
        for i in range(len(src_w)):
            pointsInRightUsingLeftCoord = self.__rotate_vac_to_angle([dst_w[i], dst_h[i]], self.right_img_sina,
                                                                     self.right_img_cosa, self.right_img_k)
            # print("bef", [dst_w[i], dst_h[i]], pointsInRightUsingLeftCoord, self.right_img_sina, self.right_img_cosa)
            # 图像中心对齐
            pointsInRightUsingLeftCoord[0] += delmid[0]
            pointsInRightUsingLeftCoord[1] += delmid[1]
            # print("aft", [dst_w[i], dst_h[i]], pointsInRightUsingLeftCoord, self.right_img_sina, self.right_img_cosa)
            self.piex_distance_w.append(round(pointsInRightUsingLeftCoord[0] - src_w[i]))
            self.piex_distance_h.append(round(pointsInRightUsingLeftCoord[1] - src_h[i]))

        # 发现边缘的像素点相对中心存在偏移，这里计算像素在图像位置上的偏移参数
        self.coef_w = np.polyfit(src_w, self.piex_distance_w, 1)
        self.coef_h = np.polyfit(src_h, self.piex_distance_h, 1)

        # self.piex_w = np.median(self.piex_distance_w)
        # self.piex_h = np.median(self.piex_distance_h)
        self.piex_w = np.polyval(self.coef_w, self.leftMedia[0])
        self.piex_h = np.polyval(self.coef_h, self.leftMedia[1])

        # 绘制像素差与像素坐标的关系图
        # y_fit = np.polyval(self.coef_w, src_w)
        # plt.plot(src_w, y_fit)
        # plt.show()
        # y_fit = np.polyval(self.coef_h, src_h)
        # plt.plot(src_h, y_fit)
        # plt.show()

        # 计算比例系数和角度
        # 计算两个向量的坐标
        gps_w = self.rightGPS[0] * self.zoom - self.leftGPS[0] * self.zoom
        gps_h = self.rightGPS[1] * self.zoom - self.leftGPS[1] * self.zoom

        self.piex_k, self.piex_sina, self.piex_cosa = self.__calculate_angle_k_of_vactor([gps_w, gps_h],
                                                                                         [self.piex_w, self.piex_h])

        print("-----sift: 特征点寻找完毕，共", len(self.src_result_pts), "对", "gps差：", [gps_w, gps_h], "像素差:",
              [self.piex_w, self.piex_h])
        # print("宽度方向特征点像素差:", self.piex_w, " ", self.piex_distance_w, "高度方向特征点像素差:", self.piex_h, " ",self.piex_distance_h)
        # test = self._drawMatches(self.__imgLeft, self.__imgRight, self.src_result_pts, self.dst_result_pts)
        # test = cv2.resize(test, (1800, 900))
        # cv2.imshow("t" + str(self.rightGPS[0]), test)

    # 进行坐标转换
    def getGPS(self, w, h, img_sig='l'):
        if img_sig == 'r':
            [w, h] = self.__rotate_vac_to_angle([w, h], self.right_img_sina, self.right_img_cosa,
                                                self.right_img_k)
            w = w - self.piex_w
            h = h - self.piex_h
        w = w - np.polyval(self.coef_w, w) + self.piex_w
        h = h - np.polyval(self.coef_h, h) + self.piex_h
        piex_coord = [w - self.leftMedia[0],
                      h - self.leftMedia[1]]
        gps_coord = self.__rotate_vac_to_angle(piex_coord, self.piex_sina, self.piex_cosa, self.piex_k)
        gps_coord = [gps_coord[0] / self.zoom + self.leftGPS[0], gps_coord[1] / self.zoom + self.leftGPS[1]]

        return gps_coord

    def mergeImages(self, filename="result"):
        """
        进行图像拼接
        :param filename:保存文件名
        :return:
        """
        if self.src_result_pts is None:
            self.computeImages()
        # 拼接图像
        self.imgFinal = self._getFinalImg(self.__H)
        cv2.imwrite(filename + ".png", self.imgFinal)

        if self.showImg:
            test = self._drawMatches(self.__imgLeft, self.__imgRight, self.src_result_pts, self.dst_result_pts)
            test = cv2.resize(test, (1800, 900))
            cv2.imshow("test", test)
            showimg = cv2.resize(self.imgFinal, (1800, 900))
            cv2.imshow("final", showimg)
            cv2.waitKey(0)


# 读取文件exif信息
def getGPS(file):
    """
    这里要注意，opencv左上角为原点，w，h为相对原点的宽、长度距离
    """
    gpsListLong = None
    gpsListLa = None
    gps = None
    with open(file, 'rb') as f:
        info = exifread.process_file(f)
        try:
            gpsListLong = info.get('GPS GPSLongitude', '0').values
            gpsListLa = info.get('GPS GPSLatitude', '0').values
            gps = GPSList2Float(gpsListLong, gpsListLa)
        except:
            gps = getGPSfromCsv('./data2/', file.split('_')[0].split('/')[-1], 0, 6, 4, 100)
    # print("gps:", gps)
    return gps


def getGPSfromCsv(path, file_number, file_number_column, lon_column, lat_column, zoom):
    global GPSCsv
    if GPSCsv is None:
        files = os.listdir(path)
        file = None
        for f in files:
            if f.endswith('.csv'):
                file = f
        if file is None:
            raise RuntimeError("!!!!!" + file_number + " Image doesn't have GPS attributes!!!!!")
        # 读取csv文件
        with open(os.path.join(path, file)) as f:
            csv_f = csv.reader(f)
            GPSCsv = []
            for row in csv_f:
                GPSCsv.append(row)
    lon = None
    lat = None
    for row in GPSCsv:
        if row[file_number_column] == file_number:
            lon = float(row[lon_column]) / zoom
            lat = float(row[lat_column]) / zoom
            print("-----loaded image ", file_number, " gps from csv:", lon, ',', lat)
    return [lon, lat]


# gps度分秒转小数
def GPSList2Float(gpsListLong, gpsListLa):
    if gpsListLong is None or gpsListLa is None: return None
    return [gpsListLong[0] + gpsListLong[1] / 60 + float(gpsListLong[2] / 3600),
            gpsListLa[0] + gpsListLa[1] / 60 + float(gpsListLa[2] / 3600)]


# gps小数转度分秒
def GPSFloat2List(gpsLong, gpsLa):
    def get_DuFenMiao(num):
        du = math.floor(num)
        num = (num - du) * 60
        fen = math.floor(num)
        num = (num - fen) * 60
        miao = fractions.Fraction(num)
        return [du, fen, miao]

    return [get_DuFenMiao(gpsLong), get_DuFenMiao(gpsLa)]


def test_images(txt_name, img1_url, img2_url, img3_url, task_sum, multiple=False, print_lock=None, txt_lock=None,
                process=0, pocess_lock=None):
    def write_list(file, data):
        for i in data:
            file.write(str(i) + '\n')

    imgL = Image.open(img1_url)
    imgM = Image.open(img2_url)
    imgR = Image.open(img3_url)

    # imgMGPS = [119.4223702, 28.5220818]
    # imgLGps = [119.422124, 28.5220846]
    # imgRGPS = [119.4226155, 28.5220789]

    siftImageOperatorML = SiftImageOperator(imgM, imgL, getGPS(img2_url), getGPS(img1_url), False)
    # siftImageOperatorML = SiftImageOperator(imgM, imgL, imgMGPS, imgLGps, False)
    siftImageOperatorML.computeImages()
    siftImageOperatorMR = SiftImageOperator(imgM, imgR, getGPS(img2_url), getGPS(img3_url), False)
    # siftImageOperatorMR = SiftImageOperator(imgM, imgR, imgMGPS, imgRGPS, False)
    siftImageOperatorMR.computeImages()
    common_points = []

    # 在中间图上绘制共同的特征点，以及计算共同特征点在三图间的误差
    def draw_points_on_image():
        img = cv2.imread(img2_url)
        for left_i, left_p in enumerate(siftImageOperatorML.src_result_pts):
            for right_i, right_p in enumerate(siftImageOperatorMR.src_result_pts):
                if round(left_p[0][0]) == round(right_p[0][0]) and round(left_p[0][1]) == round(right_p[0][1]):
                    common_points.append([round(x) for x in left_p[0]])
                    img = cv2.circle(img, tuple(common_points[-1]), 5, (255, 0, 0), 2)
        img = cv2.resize(img, (1800, 900))
        cv2.imshow("common_ponits", img)

        delta_gps = []
        for point in common_points:
            gps_left = siftImageOperatorML.getGPS(point[0], point[1])
            gps_right = siftImageOperatorMR.getGPS(point[0], point[1])
            delta_gps.append([gps_left[0] - gps_right[0], gps_left[1] - gps_right[1]])

    # 计算模型误差
    def compute_diff():
        return [str(100 * abs(1 - abs(siftImageOperatorML.piex_k / siftImageOperatorMR.piex_k))),
                str(100 * abs(1 - abs(siftImageOperatorML.piex_cosa / siftImageOperatorMR.piex_cosa)))]

    # 文件输出模型的对比信息
    def print_txt():
        txt = open(txt_name, 'a')
        txt.write(img1_url.split('/')[-1] + ' ' + img3_url.split('/')[-1] + ' 匹配度(%):k' + compute_diff()[0] + "   a:" +
                  compute_diff()[1] + '\n')
        txt.write(" delta distance:" + str((((siftImageOperatorML.getGPS(0, 0)[0] - siftImageOperatorMR.getGPS(0, 0)[
            0]) * 111000) ** 2 + ((siftImageOperatorML.getGPS(0, 0)[1] - siftImageOperatorMR.getGPS(0, 0)[
            1]) * 111000 * 2 / 3) ** 2) ** 0.5) + '\n')
        txt.write('-----left img warp:' + str(
            siftImageOperatorML.right_img_sina) + str(
            siftImageOperatorML.right_img_k) + ' right img warp:' + str(siftImageOperatorMR.right_img_sina) + str(
            siftImageOperatorMR.right_img_k) + '\n')
        txt.write('gps:' + str(siftImageOperatorML.rightGPS[0] * siftImageOperatorML.zoom - siftImageOperatorML.leftGPS[
            0] * siftImageOperatorML.zoom) + ','
                  + str(siftImageOperatorML.rightGPS[1] * siftImageOperatorML.zoom - siftImageOperatorML.leftGPS[
            1] * siftImageOperatorML.zoom) + '   piex:' + str(siftImageOperatorML.piex_w) + ',' + str(
            siftImageOperatorML.piex_h) + '\n')
        txt.write('gps:' + str(siftImageOperatorMR.rightGPS[0] * siftImageOperatorMR.zoom - siftImageOperatorMR.leftGPS[
            0] * siftImageOperatorMR.zoom) + ','
                  + str(siftImageOperatorMR.rightGPS[1] * siftImageOperatorMR.zoom - siftImageOperatorMR.leftGPS[
            1] * siftImageOperatorMR.zoom) + '   piex:' + str(siftImageOperatorMR.piex_w) + ',' + str(
            siftImageOperatorMR.piex_h) + '\n')
        txt.close()

    if multiple and txt_lock is not None:
        txt_lock.acquire()
        print_txt()
        txt_lock.release()
    else:
        print_txt()

    # 同步进度
    if pocess_lock is None:
        process += 1
    else:
        process.value += 1

    # 屏幕输出模型的k和cos
    def print_model():
        if pocess_lock is None:
            print("process:", process + 1, '/', task_sum)
        else:
            print("process:", process.value + 1, '/', task_sum)
        print(img1_url.split('/')[-1] + ' ' + img3_url.split('/')[-1] + ' 匹配度(%):')
        print('---------k:', siftImageOperatorML.piex_k, siftImageOperatorMR.piex_k, compute_diff()[0])
        print('-----cos a:', siftImageOperatorML.piex_cosa, siftImageOperatorMR.piex_cosa, compute_diff()[1])
        print('-----left img warp:', siftImageOperatorML.right_img_sina, siftImageOperatorML.right_img_k)
        print('-----right img warp:', siftImageOperatorMR.right_img_sina, siftImageOperatorMR.right_img_k)
        print("delta distance:",
              (((siftImageOperatorML.getGPS(0, 0)[0] - siftImageOperatorMR.getGPS(0, 0)[0]) * 111000) ** 2
               + ((siftImageOperatorML.getGPS(0, 0)[1] - siftImageOperatorMR.getGPS(0, 0)[
                          1]) * 111000 * 2 / 3) ** 2) ** 0.5)

    if multiple and print_lock is not None:
        print_lock.acquire()
        print_model()
        print_lock.release()
    else:
        print_model()


def test():
    txt_name = str(time.asctime(time.localtime(time.time())))
    txt_name = txt_name.replace(':', '-') + '.txt'
    print_lock = mp.Manager().Lock()  # 控制台输出锁
    txt_lock = mp.Manager().Lock()  # 文本输出锁
    process_lock = mp.Manager().Lock()  # 进程同步锁
    process_value = mp.Manager().Value('i', 0)  # 进度变量
    pool = mp.Pool(8)

    def get_full_img_name(num):
        return img_url_bef + str(num) + img_url_aft

    # 创建进程
    for i, data in enumerate(data_pairs):
        left_file = get_full_img_name(data[0])
        median_file = get_full_img_name(data[1])
        right_file = get_full_img_name(data[2])
        # test_images(txt_name, left_file, median_file, right_file, len(data_pairs))
        pool.apply_async(test_images, (
            txt_name, left_file, median_file, right_file, len(data_pairs), True, print_lock, txt_lock, process_value,
            process_lock))

    # 启动线程
    pool.close()
    pool.join()

    print("运行完成，记录文件名为：", txt_name)


def write_list(file, data):
    for i in data:
        file.write(str(i) + '\n')


if __name__ == '__main__':
    # mp.set_start_method("spawn")
    img_url_bef = "./data2/"
    img_url_aft = "_RGB.tif"
    # model = SiftImageOperator(Image.open(img_url_bef + str(117) + img_url_aft),Image.open(img_url_bef + str(118) + img_url_aft), [10, 10], [10, 10])
    # model.computeImages()
    # [118, 119, 120]
    # data_pairs = [[88, 89, 90], [90, 91, 92], [93, 94, 95], [96, 97, 98], [99, 100, 101], [102, 103, 104],
    #              [105, 106, 107], [108, 109, 110], [111, 112, 113], [114, 115, 116], [117, 118, 119], [120, 121, 122],
    #              [123, 124, 125]]
    data_pairs = [[210, 211, 212], [211, 212, 213]]
    test()
