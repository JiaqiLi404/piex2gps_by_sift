# @Time : 2022/7/3 17:03 
# @Author : Li Jiaqi
# @Description :保证疫木的唯一性工具类
import cv2
import numpy as np
import os
import csv

from datas import Config


def __getSiftFuture(image):
    grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=1000, contrastThreshold=0.001, edgeThreshold=100)
    img_copy = grayImg.copy()
    keyPoints, describes = sift.detectAndCompute(img_copy, None)
    return describes


def compareFutures(desc1, desc2):
    # K-D tree建立索引方式的常量参数
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=70)  # checks指定索引树要被遍历的次数
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches_1 = flann.knnMatch(desc1, desc2, k=2)  # 进行匹配搜索，参数k为返回的匹配点对数量
    # 把保留的匹配点放入good列表
    good1 = []
    T = Config.FUTURE_THRESHOLD  # 阈值
    # 筛选特征点
    for m, n in matches_1:
        if m.distance < T * n.distance:  # 如果最近邻点的距离和次近邻点的距离比值小于阈值，则保留最近邻点
            good1.append(m)
        #  双向交叉检查方法
    matches_2 = flann.knnMatch(desc1, desc2, k=2)  # 进行匹配搜索
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
    return good1


class Tree:
    """
    唯一树木类，包括该树木特征图，gps坐标和相同树木图像数量
    """

    def __init__(self, image, future, gps, conf):
        self.image = image
        self.future = future
        self.gps = gps
        self.conf = conf
        self.treeNum = 1
        self.gpsList = [[gps, conf]]

    def checkSimilarity(self, future):
        """
        判断该树木是否与本类疫木一致,越小越相似
        :param future:
        :return: int 相似度
        """
        # return np.sum(np.abs(np.subtract(future, self.future)))
        return 200 * len(compareFutures(self.future, future)) / (len(self.future) + len(future))

    def addTree(self, future, gps, conf):
        """
        发现一颗相同树木图像时，重新校准图像特征和gps
        :param future: 图像特征
        :param gps: gps
        """
        self.treeNum += 1
        # 12用于不同树数，不同权重
        a_1 = (self.treeNum - 1) / self.treeNum
        a_2 = 1 - a_1
        # 34用于不同置信度不同权重
        a_sum = np.exp(conf) + np.exp(self.conf)
        a_3 = np.exp(self.conf) / a_sum
        a_4 = np.exp(conf) / a_sum

        self.gps = [self.gps[0] * a_3 + gps[0] * a_4, self.gps[1] * a_3 + gps[1] * a_4]
        self.conf = max(conf, self.conf)
        self.gpsList.append([gps, conf])
        # self.future = np.array(self.future, dtype=np.float32)
        # self.future = a_1 * self.future + a_2 * future
        # self.future = np.array(np.round(self.future), dtype=np.uint8)


class UniqueTreeSets:
    def __init__(self):
        self.uniqueTreeNum = 0
        self.treeImageNum = 0
        self.trees = []

    def getNearbyTreesByGPS(self, lon, lat):
        """
        获取可能重复的疫木，根据距离
        :param lon:
        :param lat:
        :return:[Tree1,Tree2...]
        """
        res = []
        for tree in self.trees:
            distance = abs(tree.gps[0] + tree.gps[1] - lon - lat) / 1.5
            if distance < Config.TOO_CLOSE_DISTANCE / 1e5:
                res.append(tree)
        return res

    def checkIfSame(self, tree, future):
        """
        判断两棵树是否一致
        :return:
        """
        return tree.checkSimilarity(future)

    def addTree(self, tree, future, gps, conf):
        """
        将一棵树插入到已有树中
        :param tree: 已有树
        :param future: 当前树特征
        :param gps: 当前树gps
        """
        tree.addTree(future, gps, conf)
        self.treeImageNum += 1

    def addUniqueTree(self, image, future, gps, conf):
        """
        将一棵树新加到树集合中
        :param future: 当前树特征
        :param gps: 当前树gps
        """
        self.treeImageNum += 1
        self.uniqueTreeNum += 1
        self.trees.append(Tree(image, future, gps, conf))

    def saveUniqueTrees(self):
        """
        将所有唯一树图像保存下来，返回疫木的gps
        :return: res [gps1,gps2...]
        """
        res = []
        allGpsRes = []
        for tree in self.trees:
            # 重新校准各树的gps，将gps和置信度存为csv
            # 得到总权重
            a_sum = 0
            for gpsset in tree.gpsList:
                conf = gpsset[1]
                a_sum += np.exp(conf)
            # 计算gps
            tree.gps = [0, 0]
            for gpsset in tree.gpsList:
                gps = gpsset[0]
                conf = gpsset[1]
                tree.gps[0] += np.exp(conf) / a_sum * gps[0]
                tree.gps[1] += np.exp(conf) / a_sum * gps[1]
            gpsRes = [tree.gps[0], tree.gps[1]]
            for gpsset in tree.gpsList:
                gpsRes.extend([gpsset[0][0], gpsset[0][1], gpsset[1]])
            cv2.imwrite(os.path.join(Config.ILL_TREES_IMAGE_PATH, str(tree.gps[0]) + str(tree.gps[1]) + '.jpg'),
                        tree.image)
            res.append(tree.gps)
            allGpsRes.append(gpsRes)
        with open('datas/tree_all_gps_debug.csv', mode='w', newline='') as f:
            csv_f = csv.writer(f)
            for row in allGpsRes:
                csv_f.writerow(row)

        return res


def convertImageToFutureImg(image):
    """
    将图像转为特征图
    :param image: 树木图像
    :return: future
    """
    # img = image.copy()
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # img = img[:, :, 0]
    # img = np.array(img, dtype=np.uint8)
    # future = cv2.resize(img, (64, 64))
    # return future
    return __getSiftFuture(image)


def loadImageFuture(imgId, boxId):
    return np.load('tempData/illTreeFeatures/' + str(imgId) + '.' + str(boxId) + '.npy')
