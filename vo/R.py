# @Time : 2022/5/21 22:24 
# @Author : Li Jiaqi
# @Description : 标准返回文件
import enums.ResultCode as resCode


class R:
    def __init__(self, code, msg, data):
        self.code = code
        self.msg = msg
        self.data = data

    def toDict(self):
        return {"code": self.code, "data": self.data, "msg": self.msg}


def success(code=resCode.SUCCESS, msg=resCode.SUCCESS_DESC, data=None):
    if data is None:
        data = []
    return R(code, msg, data).toDict()


def fail(code=resCode.UNKNOWN_FAIL, msg=resCode.UNKNOWN_FAIL_DESC, data=None):
    if data is None:
        data = []
    return R(code, msg, data).toDict()
