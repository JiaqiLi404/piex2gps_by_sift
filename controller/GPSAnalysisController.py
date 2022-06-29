# @Time : 2022/5/21 21:04
# @Author : Li Jiaqi
# @Description : GPS分析Controller
from flask import Blueprint, request

import Config
import Utils.RedisUtil as redis
from service import GPSAnalysisService as gpsAnalysisService
from vo import R
import enums.ResultCode as resCode
import os
import time

# 起始URL"/gps"
gpsAnalysisApi = Blueprint('gpsAnalysis', __name__)


@gpsAnalysisApi.route('/', methods=['GET', 'POST'])
def hello():
    """
        查看运行状态
        :return: 运行中
    """
    res = gpsAnalysisService.getAnalyzeStatus()
    if res is None:
        res = "目前没有任何任务"
    return res


@gpsAnalysisApi.route('/start', methods=['GET', 'POST'])
def startImageAnalysis():
    """
        用于通知程序开始进行图片的分析
        :return:"图片成功开始分析"
    """
    try:
        gpsAnalysisService.startAnalyzeImages()
    except Exception as e:
        return R.fail(code=resCode.ILLEGAL_DATA, msg="图片解析失败", data=repr(e))
    return R.success(msg="图片成功开始分析")


@gpsAnalysisApi.route('/end', methods=['GET', 'POST'])
def endImageAnalysis():
    """
        用于通知程序结束图片的分析
        :return:"任务结束成功"
    """
    try:
        gpsAnalysisService.endAnalyzeImages()
    except RuntimeWarning:
        return R.success(msg="目前没有任务在运行")
    except Exception as e:
        return R.fail(code=resCode.UNKNOWN_FAIL, msg="任务结束失败", data=repr(e))
    return R.success(msg="任务结束成功")


@gpsAnalysisApi.route('/submitDet', methods=['POST'])
def submitDetection():
    file = request.files.get("result")
    if file is None:
        return R.fail(resCode.REQUEST_PARAMS_ILLEGAL, "检测结果文件未传输")
        # 直接使用文件上传对象保存
    fileName = str(time.asctime(time.localtime(time.time())))
    fileName = fileName.replace(':', '-') + '.csv'
    file.save(os.path.join(Config.DETECTION_RECEIVE_PATH, fileName))
    res=gpsAnalysisService.getGPSfromCSV(fileName)
    return R.success(data=res)
