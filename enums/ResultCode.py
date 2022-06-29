# @Time : 2022/5/22 0:25 
# @Author : Li Jiaqi
# @Description : 返回码统一文件
"""
结果码枚举类，设计参考http的返回码
200开头：表示成功
401开头：表示无访问权限
403开头：表示无操作权限
404开头：表示xx资源不存在，如单据不存在等
405开头：表示请求参数非法等异常
406开头：表示业务异常，不符合业务规则等
500开头：表示服务端内部异常
03开头：表示服务不可用
"""
# 操作成功200
SUCCESS = '20000000'
SUCCESS_DESC = '操作成功/Success'

# 无访问权限401
NOT_SIGN_IN = '40100001'  # 123
NOT_SIGN_IN_DESC = "未登录或登录过期/Not sign in"
NO_AUTH_ACCESS = '40400002'
NO_AUTH_ACCESS_DESC = "用户无访问权限/Illegal authorization"

# 无操作权限403
NO_AUTH_OPERATE = '40300000'
NO_AUTH_OPERATE_DESC = '用户无操作权限/Illegal authorization'

# 资源不存在404
DATA_NOT_EXIST = '40400000'
DATA_NOT_EXIST_DESC = '数据不存在/Data is not exist'

# 请求方参数非法405
REQUEST_PARAMS_ILLEGAL = '40500000'
REQUEST_PARAMS_ILLEGAL_DESC = '请求方参数非法/Illegal request params'

# 业务异常406
ILLEGAL_DATA = '40500000'
ILLEGAL_DATA_DESC = "非法数据/Illegal data"

# 系统异常500
UNKNOWN_FAIL = '500'
UNKNOWN_FAIL_DESC = "未知错误/Unknown error"
