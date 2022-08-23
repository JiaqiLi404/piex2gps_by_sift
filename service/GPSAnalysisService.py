# @Time : 2022/5/21 21:04
# @Author : Li Jiaqi
# @Description : 图像GPS映射参数的获得
import csv
import time

import cv2

from Utils.SiftUtil import SiftImageOperator
from PIL import Image
from datas import Config
import os
import multiprocessing as mp
import Utils.GPSUtil as gpsUtil
import Utils.RedisUtil as redis
import numpy as np
import matplotlib.pyplot as plt
import Utils.UniqueTreeSetUtil as treeUtil

pool = None


def getAnalyzeStatus():
    if pool is None:
        return None
    return "运行中"


def startAnalyzeImages():
    """
    开始进行多进程的图片分析程序，获取其gps映射函数并存入redis
    :return:
    """
    global pool

    # 清除缓存
    redis.clearAll()

    imageNumNameMap = {}
    imagePairs = []
    try:
        for image in os.listdir(Config.DATA_PATH):
            if image.endswith('TIF') or image.endswith('tif') or image.endswith('JPG') or image.endswith(
                    'jpg') or image.endswith('PNG') or image.endswith('png'):
                imageNumNameMap[__getImageNumFromName(image)] = os.path.join(Config.DATA_PATH, image)
    except:
        print("读取错误")
    # 按照数字大小排序图片文件，而不是字典序，随后生成图片对
    imgs = sorted(imageNumNameMap.keys())
    imagePair = []
    for key in imgs:
        if len(imagePair) == 0:
            imagePair.append(key)
        else:
            imagePair.append(key)
            imagePairs.append(imagePair.copy())
            imagePair = []
    if len(imagePair) == 1:
        imagePair.append(imgs[-2])
        imagePairs.append(imagePair.copy())
    # imagePair = [imageNumNameMap[imgs[0]]]
    # for key in imgs[1:]:
    #     imagePair.append(imageNumNameMap[key])
    #     imagePairs.append(imagePair.copy())
    #     imagePair = [imageNumNameMap[key]]

    # 多进程处理图片
    pool = mp.Pool(Config.PROCESS_NUM)
    csvReader = gpsUtil.CSVReader()
    csvReader.readCSV(Config.DATA_PATH)

    __analyzeImagePair(imageNumNameMap, imagePairs[0][0], imagePairs[0][1], csvReader, 0)
    for i, pair in enumerate(imagePairs):
        if i == 0:
            pool.apply_async(__analyzeImagePair,
                             (imageNumNameMap, pair[0], pair[1], csvReader, 0))
        elif i == len(imagePairs) - 1:
            pool.apply_async(__analyzeImagePair,
                             (imageNumNameMap, pair[0], pair[1], csvReader, -1,))
        else:
            pool.apply_async(__analyzeImagePair, (
                imageNumNameMap, pair[0], pair[1], csvReader, i,))

    pool.close()
    # pool.join()
    print("***** calculating ", len(imagePairs), " image pairs")


def endAnalyzeImages():
    """
    结束进程池，结束任务
    :return:
    """
    global pool
    if pool is None:
        raise RuntimeWarning("没有任务在工作/No task is running")
    pool.terminate()
    pool = None
    print("任务结束成功")


def getGPSfromCSV(fileName):
    """
    将疫木结果解析出gps
    :param fileName:
    :return:
    """
    global pool
    pool = mp.Pool(Config.PROCESS_NUM)

    allModelDicts = __calculateConfAfterSiftModel()
    csvReader = gpsUtil.CSVReader()
    results = csvReader.getDetectionResultfromCsv(os.path.join(Config.DETECTION_RECEIVE_PATH, fileName))
    results.sort(key=lambda ele: ele[0])
    treeSet = treeUtil.UniqueTreeSets()
    __calculateSiftFutureOfBNDBoxes(results)
    for resPerImage in results:
        # [num,box1,box2..] 疫木结果box
        imageNum = resPerImage[0]
        model_dict = redis.getDict(imageNum)
        # # 如果没该图片的模型参数，则报错
        # if not model_dict:
        #     raise RuntimeError("-----model ", imageNum,
        #                        " is empty, maybe its image file is not exists, please check manually")

        for index, resPerBox in enumerate(resPerImage[1:]):
            # 每个疫木框
            [w1, h1, w2, h2, lon, lat] = resPerBox
            # 去除重复点
            image = Image.open(os.path.join(Config.DATA_PATH, __getImageNameFromNum(imageNum)))
            model = SiftImageOperator.gen_from_dict(image, image, model_dict)
            cvimage = cv2.imread(os.path.join(Config.DATA_PATH, __getImageNameFromNum(imageNum)))
            treeImage = cvimage[int(h1): int(h2), int(w1): int(w2)]
            treeFeature = treeUtil.loadImageFuture(imageNum, index)
            treeCenter = [(w1 + w2) / 2, (h1 + h2) / 2]
            gpsRes = model.getGPS(treeCenter[0], treeCenter[1], model_dict[b"image_pos"])
            addedFlag = False
            for tree in treeSet.getNearbyTreesByGPS(gpsRes[0], gpsRes[1]):
                if Config.COMMON_SHOW_WINDOW:
                    future = cv2.cvtColor(treeImage, cv2.COLOR_BGR2HSV)
                    future = future[:, :, 0]
                    future = np.array(future, dtype=np.uint8)
                    a = treeSet.checkIfSame(tree, treeFeature)
                    cv2.destroyAllWindows()
                    cv2.imshow('same1:' + str(a), tree.image)
                    cv2.imshow('same2:' + str(a), treeImage)
                    cv2.imshow('future:' + str(a), future)
                    cv2.resizeWindow('same1:' + str(a), 300, 300)
                    cv2.resizeWindow('same2:' + str(a), 300, 300)
                    cv2.resizeWindow('future:' + str(a), 300, 300)
                    cv2.moveWindow('same1:' + str(a), 0, 0)
                    cv2.moveWindow('same2:' + str(a), 400, 0)
                    cv2.moveWindow('future:' + str(a), 800, 0)
                    print('?????test :similarity:', str(a), 'futureNum:', len(tree.future), ' ', len(treeFeature))

                    cv2.waitKey(2000)
                if treeSet.checkIfSame(tree, treeFeature) > Config.COMMON_LIMIT:
                    treeSet.addTree(tree, treeFeature, gpsRes, float(model_dict[b'confidence']))
                    addedFlag = True
                    break
            if not addedFlag and float(model_dict[b'confidence'])>0.1:
                treeSet.addUniqueTree(treeImage, treeFeature, gpsRes, float(model_dict[b'confidence']))

    gpsResults = treeSet.saveUniqueTrees()

    try:
        endAnalyzeImages()
    except RuntimeWarning:
        pass
    # 保存gps列表
    con = __saveGPSResult(Config.ALL_GPS_RESULT_PATH, gpsResults)
    finalResults = gpsResults

    # finalResults = []
    # # 去除重复的点
    # for i, v1 in enumerate(gpsResults):
    #     flag = True
    #     for j, v2 in enumerate(gpsResults[i + 1:]):
    #         distance = __calcalateDistancefromGPS(v1, v2)
    #         if distance < Config.TOO_CLOSE_DISTANCE:
    #             flag = False
    #     if flag:
    #         finalResults.append(v1)
    #
    # con = __saveGPSResult(Config.FINAL_GPS_RESULT_PATH, finalResults)
    print("***** model calculated ", treeSet.treeImageNum, " trees, ", treeSet.uniqueTreeNum, " trees are unique")

    return con


def unique(all_points_file):
    """
    判断所有点内的之间距离
    :param all_points_file:
    :return:
    """
    finalResults = []
    gpsResults = []
    with open(os.path.join(Config.ALL_GPS_RESULT_PATH, all_points_file)) as f:
        csv_f = csv.reader(f)
        for row in csv_f:
            gpsResults.append([float(row[0]), float(row[1])])

    # 去除重复的点
    minds = []
    for i, v1 in enumerate(gpsResults):
        flag = True
        mind = 1e8
        for j, v2 in enumerate(gpsResults[i + 1:]):
            distance = __calcalateDistancefromGPS(v1, v2)
            if (distance < mind):
                mind = distance
            if distance < Config.TOO_CLOSE_DISTANCE:
                flag = False
        if flag:
            finalResults.append(v1)
        if mind < 30:
            minds.append(mind)
    minds = minds[:-1]
    print(minds)
    plt.hist(minds, bins=100)
    plt.show()


######################################################################
#                            下面是私有方法                             #
######################################################################
def __calculateSiftFutureOfBNDBoxes(results):
    global pool
    for resPerImage in results:
        # [num,box1,box2..] 疫木结果box
        imageNum = resPerImage[0]

        for index, resPerBox in enumerate(resPerImage[1:]):
            pool.apply_async(__calculateSiftFutureOfBox, (resPerBox, imageNum, index,))
    pool.close()
    pool.join()


def __calculateSiftFutureOfBox(box, imgId, boxId):
    # 每个疫木框
    [w1, h1, w2, h2, lon, lat] = box
    # 去除重复点
    cvimage = cv2.imread(os.path.join(Config.DATA_PATH, __getImageNameFromNum(imgId)))
    treeImage = cvimage[int(h1): int(h2), int(w1): int(w2)]
    treeFeature = treeUtil.convertImageToFutureImg(treeImage)
    np.save('tempData/illTreeFeatures/' + str(imgId) + '.' + str(boxId) + '.npy', treeFeature)


def __calculateConfAfterSiftModel():
    # 计算置信度
    # 拿到所有的图片序号
    imageNumNameMap = {}
    for image in os.listdir(Config.DATA_PATH):
        if image.endswith('TIF') or image.endswith('tif') or image.endswith('JPG') or image.endswith(
                'jpg') or image.endswith('PNG') or image.endswith('png'):
            imageNumNameMap[__getImageNumFromName(image)] = os.path.join(Config.DATA_PATH, image)
    imgs = sorted(imageNumNameMap.keys())
    # 拿到所有的像素gps转换模型 和 需要着重处理的模型参数
    allModelParams = []
    allModelDicts = []
    for index, i in enumerate(imgs):
        dict = redis.getDict(i)
        # 如果没该图片的模型参数，则报错
        if not dict:
            raise RuntimeError("-----model ", i, " is empty, maybe its image file is not exists, please check manually")
        allModelDicts.append(dict)
        if index % 2 == 0:
            allModelParams.append([float(dict[b'piex_k']), float(dict[b'piex_cosa']), float(dict[b'piex_sina'])])

    # 计算置信度
    def calculateConfBetweenParams(param1, param2):
        # piex_k带来的置信度
        conf_k = min(abs(param1[0] - param2[0]) / (param1[0] + param2[0]) * 2, 1)
        # piex_a带来的置信度
        angle_1 = np.angle(param1[1] + 1j * param1[2])
        angle_2 = np.angle(param2[1] + 1j * param2[2])
        conf_angle = min(abs(angle_1 - angle_2) / (abs(angle_1) + abs(angle_2)) * 2, 1)

        return 1 - Config.GPS_EVALUATE_K_WEIGHT * conf_k - Config.GPS_EVALUATE_A_WEIGHT * conf_angle

    for i in range(int(np.round(len(allModelDicts) / 2))):
        # 根据前面的计算置信度
        conf = 0.25
        if i != 0:
            conf = 0.5 * calculateConfBetweenParams(allModelParams[i], allModelParams[i - 1])
        # 根据后面的计算置信度
        if i != int(np.round(len(allModelDicts) / 2)) - 1:
            conf += 0.5 * calculateConfBetweenParams(allModelParams[i], allModelParams[i + 1])
        else:
            conf += 0.25
        # 保证conf非0
        conf = max(1e-5, conf)
        allModelDicts[i * 2][b'confidence'] = conf
        if i * 2 + 1 < len(allModelDicts):
            allModelDicts[i * 2 + 1][b'confidence'] = conf
    # 存回redis中
    for i, dict in enumerate(allModelDicts):
        redis.setDict(imgs[i], dict)
    return allModelDicts


def __calcalateDistancefromGPS(gps1, gps2):
    return (((gps1[0] - gps2[0]) * 111000) ** 2 + ((gps1[1] - gps2[1]) * 111000 * 2 / 3) ** 2) ** 0.5


def __saveGPSResult(path, gpsResults):
    """
    将疫木坐标存储为CSV文件
    :param gpsResults: 疫木坐标
    :return: 疫木坐标
    """
    txt_name = str(time.asctime(time.localtime(time.time())))
    txt_name = txt_name.replace(':', '-') + '.csv'
    con = ""
    with open(os.path.join(path, txt_name), 'a', newline='') as f:
        writer = csv.writer(f, dialect='excel')
        for row in gpsResults:
            writer.writerow(row)
            con += str(row[0]) + "," + str(row[1]) + '/n'
    return con


def __getImageNumFromName(name):
    """
    将文件从名字转为编号
    :param name: 文件名字
    :return: 文件编号
    """
    return int(name[len(Config.DATA_PRENAME):-len(Config.DATA_AFTNAME)])


def __getImageNameFromNum(num):
    """
    将文件从编号转为名字
    :param num: 文件编号
    :return: 文件名字
    """
    return Config.DATA_PRENAME + str(num) + Config.DATA_AFTNAME


# (imageNumNameMap,pair[0], pair[1], csvReader, True, None,imgLocks[i],imgLocks[1])
def __analyzeImagePair(imageNumNameMap, mainImageId, subImageId, csvReader, index):
    """
    成对处理图片，获取其gps映射函数并存入redis
    :param mainImageURL:
    :param subImageURL:
    :return:
    """
    mainImageURL = imageNumNameMap[mainImageId]
    subImageURL = imageNumNameMap[subImageId]
    mainImage = Image.open(mainImageURL)
    subImage = Image.open(subImageURL)
    mainGps = gpsUtil.getGPSfromFile(mainImageURL)
    subGps = gpsUtil.getGPSfromFile(subImageURL)
    if mainGps is None or subGps is None:
        mainGps = csvReader.getImageGPSfromCsv(Config.DATA_PATH, str(mainImageId),
                                               Config.CSV_IMAGE_GPS_READ_PARAMS[0],
                                               Config.CSV_IMAGE_GPS_READ_PARAMS[1],
                                               Config.CSV_IMAGE_GPS_READ_PARAMS[2],
                                               Config.CSV_IMAGE_GPS_READ_PARAMS[3])
        subGps = csvReader.getImageGPSfromCsv(Config.DATA_PATH, str(subImageId),
                                              Config.CSV_IMAGE_GPS_READ_PARAMS[0],
                                              Config.CSV_IMAGE_GPS_READ_PARAMS[1],
                                              Config.CSV_IMAGE_GPS_READ_PARAMS[2],
                                              Config.CSV_IMAGE_GPS_READ_PARAMS[3])
    # try:
    model = SiftImageOperator(mainImage, subImage, mainGps, subGps, False, nfeatures=Config.SIFT_N_FEATURES)
    model.computeImages()
    # except:
    #     print("模型计算错误")
    dictl = model.to_dict()
    dictr = model.to_dict()
    dictl["image_pos"] = 'l'
    dictr["image_pos"] = 'r'
    redis.setDict(mainImageId, dictl)
    if subImageId > mainImageId:
        redis.setDict(subImageId, dictr)
    print("***** sift: a task is succeed,left:", mainImageId, 'right:', subImageId)


def __score_gps_result():
    pass


def __saveIllnessTree(image, w1, h1, w2, h2, gpsRes):
    cv2.imwrite(os.path.join(Config.ILL_TREES_IMAGE_PATH, str(gpsRes[0]) + str(gpsRes[1]) + '.jpg'),
                image[h1:h2, w1:w2])

######################################################################
#                          下面是被封印的代码                            #
######################################################################
# if __name__ == '__main__':
#     imageNumNameMap = {}
#     imagePairs = []
#     for image in os.listdir(Config.DATA_PATH):
#         if image.endswith('TIF') or image.endswith('tif') or image.endswith('JPG') or image.endswith(
#                 'jpg') or image.endswith('PNG') or image.endswith('png'):
#             imageNumNameMap[__getImageNumFromName(image)] = os.path.join(Config.DATA_PATH, image)
#     # 按照数字大小排序图片文件，而不是字典序，随后生成图片对
#     imgs = sorted(imageNumNameMap.keys())
#     imagePair = []
#     for key in imgs:
#         if len(imagePair) == 0:
#             imagePair.append(key)
#         else:
#             imagePair.append(key)
#             imagePairs.append(imagePair.copy())
#             imagePair = []
#     if len(imagePair) == 1:
#         imagePair.append(imageNumNameMap[imgs[-2]])
#         imagePairs.append(imagePair.copy())
#
#     # 处理图片
#     pool = mp.Pool(Config.PROCESS_NUM)
#     csvReader = gpsUtil.CSVReader()
#     csvReader.readCSV(Config.DATA_PATH)
#     imgLocks = []
#     for i in range(len(imagePairs)):
#         imgLocks.append(mp.Lock())
#     for i, pair in enumerate(imagePairs):
#         if i == 0:
#             __analyzeImagePair(imageNumNameMap, pair[0], pair[1], csvReader,i)
#         elif i == len(imagePairs) - 1:
#             __analyzeImagePair(imageNumNameMap, pair[0], pair[1], csvReader,-1)
#         else:
#             __analyzeImagePair(
#                 imageNumNameMap, pair[0], pair[1], csvReader,i)
#
#     pool.close()
