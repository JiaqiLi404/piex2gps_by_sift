# @Time : 2022/5/22 12:29 
# @Author : Li Jiaqi
# @Description :
import exifread
import os
import csv
import Config


class CSVReader:
    def __init__(self):
        self.GPSCsv = []

    def readCSV(self, path):
        """
        将csv内容缓存进对象
        :param path: csv所在路径
        :return: void
        """
        files = os.listdir(path)
        file = None
        for f in files:
            if f.endswith('.csv'):
                file = f
        if file is None:
            raise RuntimeError("!!!!!Image doesn't have GPS attributes!!!!!")
        # 读取csv文件
        with open(os.path.join(path, file)) as f:
            csv_f = csv.reader(f)
            self.GPSCsv = []
            for row in csv_f:
                self.GPSCsv.append(row)

    def getImageGPSfromCsv(self, path, file_number, file_number_column, lon_column, lat_column, zoom=1):
        """
        读取图像的gps
        :param path: csv所在路径
        :param file_number: 图像编号
        :param file_number_column:  图像编号所在列
        :param lon_column: 经度所在列
        :param lat_column: 纬度所在列
        :param zoom: 缩放比例
        :return:
        """
        if self.GPSCsv is None:
            self.readCSV(path)
        lon = None
        lat = None
        for row in self.GPSCsv:
            if row[file_number_column] == str(file_number):
                lon = float(row[lon_column]) / zoom
                lat = float(row[lat_column]) / zoom
                print("-----csv: loaded image ", file_number, " gps from csv:", lon, ',', lat)
        if lon is None or lat is None:
            print("!!!!!Image doesn't have GPS attributes!!!!!")
            raise RuntimeError("!!!!!Image doesn't have GPS attributes!!!!!")
        return [lon, lat]

    def getDetectionResultfromCsv(self, file):
        """
        读取疫木检测结果
        :param file: 疫木检测结果文件
        :return: 检测结果
        """
        results = []
        with open(file) as f:
            csv_f = csv.reader(f)
            # 去除标题行
            next(csv_f)
            self.GPSCsv = []
            for row in csv_f:
                self.GPSCsv.append(row)
                result = [getImageNumFromName(row[0])]
                box = []
                for i in row[1:]:
                    box.append(float(i))
                    if len(box) == 6:
                        result.append(box.copy())
                        box.clear()
                results.append(result.copy())
        return results


# 读取文件exif信息
def getGPSfromFile(file):
    """
    这里要注意，opencv左上角为原点，w，h为相对原点的宽、长度距离
    """
    gpsListLong = None
    gpsListLa = None
    gps = None
    with open(file, 'rb') as f:
        info = exifread.process_file(f)
        try:
            gpsListLong = info.get('GPS GPSLongitude', '0').values
            gpsListLa = info.get('GPS GPSLatitude', '0').values
            gps = GPSList2Float(gpsListLong, gpsListLa)
            print("-----exif: loaded gps from image:", file, ' ', gps)
        except:
            pass
    return gps


# gps度分秒转小数
def GPSList2Float(gpsListLong, gpsListLa):
    if gpsListLong is None or gpsListLa is None: return None
    return [gpsListLong[0] + gpsListLong[1] / 60 + float(gpsListLong[2] / 3600),
            gpsListLa[0] + gpsListLa[1] / 60 + float(gpsListLa[2] / 3600)]


def getImageNumFromName(name):
    """
    将文件从名字转为编号
    :param name: 文件名字
    :return: 文件编号
    """
    return int(name[len(Config.DATA_PRENAME):-len(Config.DATA_AFTNAME)])
