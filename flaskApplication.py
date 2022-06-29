from flask import *
from flask import Flask
from controller.GPSAnalysisController import gpsAnalysisApi
import Utils.RedisUtil as redis


app = Flask(__name__)  # 初始化app

# 解决浏览器输出乱码问题
app.config['JSON_AS_ASCII'] = False

# 注册Controller
# 注册 gps解析Controller
app.register_blueprint(gpsAnalysisApi, url_prefix='/gps')

if __name__ == '__main__':
    redis.clearAll()
    app.run("127.0.0.1", 5000)  # 运行app



