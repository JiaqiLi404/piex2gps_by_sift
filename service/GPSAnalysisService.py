# @Time : 2022/5/21 21:04 
# @Author : Li Jiaqi
# @Description : 图像GPS映射参数的获得
import csv
import time

from Utils.SiftUtil import SiftImageOperator, getGPS
from PIL import Image
import Config
import os
import multiprocessing as mp
import Utils.GPSUtil as gpsUtil
import Utils.RedisUtil as redis
import numpy as np
import matplotlib.pyplot as plt

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
    for image in os.listdir(Config.DATA_PATH):
        if image.endswith('TIF') or image.endswith('tif') or image.endswith('JPG') or image.endswith(
                'jpg') or image.endswith('PNG') or image.endswith('png'):
            imageNumNameMap[__getImageNumFromName(image)] = os.path.join(Config.DATA_PATH, image)
    # 按照数字大小排序图片文件，而不是字典序，随后生成图片对
    imgs = sorted(imageNumNameMap.keys())
    imagePair = [imageNumNameMap[imgs[0]]]
    for key in imgs[1:]:
        imagePair.append(imageNumNameMap[key])
        imagePairs.append(imagePair.copy())
        imagePair = [imageNumNameMap[key]]

    # 多进程处理图片
    pool = mp.Pool(Config.PROCESS_NUM)
    csvReader = gpsUtil.CSVReader()
    csvReader.readCSV(Config.DATA_PATH)
    for pair in imagePairs:
        pool.apply_async(__analyzeImagePair, (pair[0], pair[1], csvReader))
    pool.close()

    # pool.join()


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
    csvReader = gpsUtil.CSVReader()
    results = csvReader.getDetectionResultfromCsv(os.path.join(Config.DETECTION_RECEIVE_PATH, fileName))
    results.sort(key=lambda ele: ele[0])
    gpsResults = []
    for resPerImage in results:
        # [num,box1,box2..]
        imageNum = resPerImage[0]

        model_dict = redis.getDict(imageNum)
        # 如果程序还没算出该图片的模型参数，则重复获取
        while not model_dict:
            print("-----model ", imageNum, " is empty, maybe it's not calculated, retrying...")
            model_dict = redis.getDict(imageNum)
            time.sleep(Config.REDIS_REGET_DELAY)

        for resPerBox in resPerImage[1:]:
            [w1, h1, w2, h2, lon, lat] = resPerBox
            # todo:去除重复点
            image = Image.open(os.path.join(Config.DATA_PATH, __getImageNameFromNum(imageNum)))
            model = SiftImageOperator.gen_from_dict(image, image, model_dict)
            # point = [(w1 + w2) / 2, (h1 + h2) / 2]
            # gpsRes = model.getGPS(point[0], point[1], model_dict[b"image_pos"])
            # print("-----model ", imageNum, " calculated gps:", gpsRes)
            # gpsResults.append(gpsRes.copy())
            # 获取两个对角点的gps
            left_top = model.getGPS(w1, h1, model_dict[b"image_pos"])
            right_buttom = model.getGPS(w2, h2, model_dict[b"image_pos"])
            # 找到距离最近的几个目标，然后判断是否有交融，无则认为是一颗新的疫木
            gpsRes = [(left_top[0] + right_buttom[0]) / 2, (left_top[1] + right_buttom[1]) / 2]
            gpsRes.extend(left_top)
            gpsRes.extend(right_buttom)
            is_new_flag = True
            for tree in gpsResults:
                distance = abs(tree[0] + tree[1] - gpsRes[0] - gpsRes[1])
                if distance < Config.TOO_CLOSE_DISTANCE / 1e5:
                    # 求相交面积
                    ws = [left_top[0], right_buttom[0], tree[2], tree[4]]
                    hs = [left_top[1], right_buttom[1], tree[3], tree[5]]
                    w = 0
                    if ws[0] < ws[2] < ws[1]:
                        if ws[0] < ws[3] < ws[1]:
                            w = abs(ws[3] - ws[2])
                        else:
                            w = min(abs(ws[3] - ws[0]), abs(ws[3] - ws[1]))
                    elif ws[0] < ws[3] < ws[1]:
                        w = min(abs(ws[2] - ws[0]), abs(ws[2] - ws[1]))
                    h = 0
                    if hs[0] < hs[2] < hs[1]:
                        if hs[0] < hs[3] < hs[1]:
                            h = abs(hs[3] - hs[2])
                        else:
                            h = min(abs(hs[3] - hs[0]), abs(hs[3] - hs[1]))
                    elif hs[0] < hs[3] < hs[1]:
                        h = min(abs(hs[2] - hs[0]), abs(hs[2] - hs[1]))
                    area = h * w
                    # 判断是否大量重合
                    if area > Config.COMMON_AREA_LIMIT / 100 * abs(left_top[0] - right_buttom[0]) * abs(
                            left_top[1] - right_buttom[1]):
                        is_new_flag = False
                        break
            if is_new_flag:
                gpsResults.append(gpsRes.copy())

    endAnalyzeImages()
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
    print("***** model calculated ", len(gpsResults), " gps points, ", len(finalResults), " points are unique")

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
#                          下面是私有方法                               #
######################################################################
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


def __analyzeImagePair(mainImageURL, subImageURL, csvReader):
    """
    成对处理图片，获取其gps映射函数并存入redis
    :param mainImageURL:
    :param subImageURL:
    :return:
    """
    mainImage = Image.open(mainImageURL)
    subImage = Image.open(subImageURL)
    mainGps = gpsUtil.getGPSfromFile(mainImageURL)
    subGps = gpsUtil.getGPSfromFile(subImageURL)
    if mainGps is None or subGps is None:
        mainGps = csvReader.getImageGPSfromCsv(Config.DATA_PATH, mainImageURL.split('_')[0].split('\\')[-1],
                                               Config.CSV_IMAGE_GPS_READ_PARAMS[0], Config.CSV_IMAGE_GPS_READ_PARAMS[1],
                                               Config.CSV_IMAGE_GPS_READ_PARAMS[2], Config.CSV_IMAGE_GPS_READ_PARAMS[3])
        subGps = csvReader.getImageGPSfromCsv(Config.DATA_PATH, subImageURL.split('_')[0].split('\\')[-1],
                                              Config.CSV_IMAGE_GPS_READ_PARAMS[0], Config.CSV_IMAGE_GPS_READ_PARAMS[1],
                                              Config.CSV_IMAGE_GPS_READ_PARAMS[2], Config.CSV_IMAGE_GPS_READ_PARAMS[3])
    model = SiftImageOperator(mainImage, subImage, mainGps, subGps, False, nfeatures=Config.SIFT_N_FEATURES)
    model.computeImages()
    dictl = model.to_dict()
    dictr = model.to_dict()
    dictl["image_pos"] = 'l'
    dictr["image_pos"] = 'r'
    redis.setDict(__getImageNumFromName(mainImageURL.split('\\')[-1]), dictl)
    redis.setDict(__getImageNumFromName(subImageURL.split('\\')[-1]), dictr)
    print("***** sift: a task is succeed")

# unique("Sun May 22 16-18-42 2022.csv")
