# @Time : 2022/5/21 21:32 
# @Author : Li Jiaqi
# @Description : 用于存储每个gps映射模型的参数，用于存入redis，和从redis读取为对象
class SiftModelParams:
    def __init__(self, coef_w, coef_h, piex_w, piex_h, piex_k, piex_cosa, image_pos):
        """
        每个gps映射模型的参数
        :param coef_w: 参数，用于宽度方向上的边缘补偿拟合参数
        :param coef_h: 参数，用于高度方向上的边缘补偿拟合参数
        :param piex_w: 参数，用于宽度方向上的边缘补偿
        :param piex_h: 参数，用于高度方向上的边缘补偿
        :param piex_k: 参数，像素与真实gps的倍数
        :param piex_cosa: 参数，像素坐标系与真实坐标系的旋转角度
        """
        self.coef_w = coef_w
        self.coef_h = coef_h
        self.piex_h = piex_h
        self.piex_w = piex_w
        self.piex_k = piex_k
        self.piex_cosa = piex_cosa
        self.image_pos = image_pos

    def to_dict(self):
        """
        将类转为dict，方便redis存储
        :return: dict
        """
        return {"coef_w": self.coef_w, "coef_h": self.coef_h, "piex_w": self.piex_w, "piex_h": self.piex_h,
                "piex_k": self.piex_k, "piex_cosa": self.piex_cosa, "image_pos": self.image_pos}

    @classmethod
    def from_dict(self, dict):
        """
        类方法进行构造器重载，用dict构建类
        :param dict: source
        :return: SiftModelParams(dict)
        """
        return SiftModelParams(dict["coef_w"], dict["coef_h"], dict["piex_w"], dict["piex_h"], dict["piex_k"],
                               dict["piex_cosa"], dict["image_pos"])
