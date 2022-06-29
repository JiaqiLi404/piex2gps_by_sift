import os

import cv2
import numpy as np
from osgeo import gdal
from osgeo import osr


class TifOperator:
    def __init__(self, tif_file):
        self.file = gdal.Open(tif_file)
        self.XSize = self.file.RasterXSize
        self.YSize = self.file.RasterYSize
        self.gt_matrix = self.file.GetGeoTransform()
        self.proj = self.file.GetProjection()

    def get_coord(self, width, height):
        # 图上坐标转投影坐标
        x = self.gt_matrix[0] + width * self.gt_matrix[1] + height * self.gt_matrix[2]
        y = self.gt_matrix[3] + width * self.gt_matrix[4] + height * self.gt_matrix[5]
        # 投影坐标转经纬度
        prosrs = osr.SpatialReference()
        prosrs.ImportFromWkt(self.proj)
        geosrs = prosrs.CloneGeogCS()
        ct = osr.CoordinateTransformation(prosrs, geosrs)
        coords = ct.TransformPoint(x, y)
        return coords[1], coords[0]
