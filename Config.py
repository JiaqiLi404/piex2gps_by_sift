# @Time : 2022/5/21 20:56 
# @Author : Li Jiaqi
# @Description : 设置常量

"""
路径设置
"""
DATA_PATH = './data2'  # 图片数据的路径
DETECTION_RECEIVE_PATH = './illnesstreeDetectionReceive'  # 收到的疫木识别结果缓存路径
ALL_GPS_RESULT_PATH = 'allGpsResults'  # 包括重复疫木坐标的gps结果
FINAL_GPS_RESULT_PATH = './finalResults'  # 最终去除重复疫木的gps结果

"""
图片文件设置
"""
DATA_PRENAME = ''  # 图片文件的前缀 图片文件名：<前缀>编号<后缀>
DATA_AFTNAME = '_RGB.tif'  # 图片文件的前缀 图片文件名：<前缀>编号<后缀>

"""
Redis设置
"""
REDIS_HOST = "127.0.0.1"  # 主机地址
REDIS_PORT = 6379  # 主机端口
REDIS_PSWD = ""  # 主机密码
REDIS_DB = 1  # 数据库编号

REDIS_REGET_DELAY = 1  # (秒)程序还未算出参数，则需要延迟一定时间后重新从redis获取参数

"""
多进程设置
"""
PROCESS_NUM = 7  # 进程数,注意此外还需运行主线程

"""
CSV文件读取设置
"""
# 图像gps的读取，使用exif的话无需修改
CSV_IMAGE_GPS_READ_PARAMS = [0, 6, 4, 100]  # 【文件名所在列，经度所在列，纬度所在列，缩放比例】

"""
SIFT算法设置
"""
SIFT_N_FEATURES = 30000  # 特征点数目（算法对检测出的特征点排名，返回最好的nfeatures个特征点）
TOO_CLOSE_DISTANCE = 30  # 当两个疫木太近时认为可能是同一棵树
COMMON_AREA_LIMIT = 30  # 当两个矩形框有大量面积重合时认为是同一棵树(%)

"""
GPS误差图像剔除参数
"""
