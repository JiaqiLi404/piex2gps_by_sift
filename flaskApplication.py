from flask import Flask

from datas import Config
from controller.GPSAnalysisController import gpsAnalysisApi

app = Flask(__name__)  # 初始化app

# 解决浏览器输出乱码问题
app.config['JSON_AS_ASCII'] = False

# 注册Controller
# 注册 gps解析Controller
app.register_blueprint(gpsAnalysisApi, url_prefix='/gps')

if __name__ == '__main__':
    #redis.clearAll()
    app.run(Config.WEB_RUNTIME_IP, Config.WEB_RUNTIME_PORT)  # 运行app



