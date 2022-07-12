# @Time : 2022/5/21 20:56 
# @Author : Li Jiaqi
# @Description : 设置常量

"""
REST接口设置
"""
WEB_RUNTIME_IP = "0.0.0.0"  # 运行域名
WEB_RUNTIME_PORT = 5000  # 运行端口

"""
路径设置
"""
DATA_PATH = 'datas/data'  # 图片数据的路径
DETECTION_RECEIVE_PATH = 'datas/illnesstreeDetectionReceive'  # 收到的疫木识别结果缓存路径
ALL_GPS_RESULT_PATH = 'datas/allGpsResults'  # 包括重复疫木坐标的gps结果
FINAL_GPS_RESULT_PATH = 'datas/finalResults'  # 最终去除重复疫木的gps结果
SAVE_ILLTREE_IMAGE = True  # 是否存放疫木图像
ILL_TREES_IMAGE_PATH = 'datas/illnesstreeImages'  # 疫木图像存放

"""
图片文件设置
"""
DATA_PRENAME = ''  # 图片文件的前缀 图片文件名：<前缀>编号<后缀>
DATA_AFTNAME = '_RGB.tif'  # 图片文件的前缀 图片文件名：<前缀>编号<后缀>

"""
Redis设置
"""
REDIS_HOST = "172.25.160.1"  # 主机地址
REDIS_PORT = 6379  # 主机端口
REDIS_PSWD = ""  # 主机密码
REDIS_DB = 1  # 数据库编号

"""
多进程设置
"""
PROCESS_NUM = 6  # 进程数,注意此外还需运行主线程

"""
CSV文件读取设置
"""
# 图像gps的读取，使用exif的话无需修改
CSV_IMAGE_GPS_READ_PARAMS = [0, 6, 4, 100]  # 【文件名所在列，经度所在列，纬度所在列，缩放比例】

"""
SIFT算法设置
"""
SIFT_N_FEATURES = 30000  # 特征点数目（算法对检测出的特征点排名，返回最好的nfeatures个特征点）

"""
重复疫木去除参数
"""
COMMON_SHOW_WINDOW = False  # 是否展示疫木去重窗口（因有时间延迟，会影响运行效率，仅检测时使用）
TOO_CLOSE_DISTANCE = 35  # 当两个疫木太近时认为可能是同一棵树
COMMON_LIMIT = 2  # 当两个矩形框大量匹配时认为是同一棵树(%)
FUTURE_THRESHOLD = 0.77  # 特征匹配阈值，越大匹配的特征点越多，匹配效果降低；越小匹配的特征点数量较少

"""
gps结果评价参数
"""
GPS_EVALUATE_K_WEIGHT = 0.3  # 依靠相邻模型参数评价模型gps置信度中，像素向量与gps向量的缩放比例k差异对gps置信度的影响权重
GPS_EVALUATE_A_WEIGHT = 0.7  # 依靠相邻模型参数评价模型gps置信度中，像素向量与gps向量的旋转角度a差异对gps置信度的影响权重
