"""
@file:DataAnalysisExtract.py
@author: Outstanding Application Engineer
@Date:2023/8/3
@mail:Application@cn.inno.com
"""
import math
import os
from multiprocessing import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rosbag
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
import ExcelOperation
import random
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import hdbscan
import re

_DATATYPES = {}
_DATATYPES[PointField.INT8] = ('b', 1)
_DATATYPES[PointField.UINT8] = ('B', 1)
_DATATYPES[PointField.INT16] = ('h', 2)
_DATATYPES[PointField.UINT16] = ('H', 2)
_DATATYPES[PointField.INT32] = ('i', 4)
_DATATYPES[PointField.UINT32] = ('I', 4)
_DATATYPES[PointField.FLOAT32] = ('f', 4)
_DATATYPES[PointField.FLOAT64] = ('d', 8)
result = []

class Extract:

    def __init__(self):
        self.scanline = 1
        self.x = 3
        self.y = 4
        self.z = 5
        self.f = 7
        self.I = 0
        self.intensity = 6
        self.frame_count = 0
        self.fields_wanted = ['flags', 'flag', 'scan_id', 'scanline', 'scan_idx', 'x', 'y', 'z', 'intensity',
                              'reflectance', 'frame_idx', 'frame_id', 'elongation', 'is_2nd_return', 'multi_return',
                              'timestamp', 'channel', 'roi', 'facet', 'confid_level']
        self.index_sort = np.zeros((len(self.fields_wanted)), dtype=int)
        self.topics = ['/iv_points', '/AT128/pandar_points', '/rslidar_points', 'iv_points']
        self.topic = ''
        self.LiDAR_model = ['K', 'W', 'E']

    def get_pointcloud2(self, msg):
        ps = PointCloud2(header=(msg.header), height=1,
          width=(msg.row_step / msg.point_step),
          is_dense=False,
          is_bigendian=(msg.is_bigendian),
          fields=(msg.fields),
          point_step=(msg.point_step),
          row_step=(msg.row_step),
          data=(msg.data))
        return ps

    def get_struct_fmt_map(slef, is_bigendian, fields):
        """
        Get PointField from bag message using tools in ros
        """
        result = []
        fmt_pre = '>' if is_bigendian else '<'
        for field in sorted(fields, key=(lambda f: f.offset)):
            if field.datatype not in _DATATYPES:
                print('Skipping unknown PointField datatype [{}]'.format(field.datatype))
                exit(-1)
            else:
                datatype_fmt, datatype_length = _DATATYPES[field.datatype]
                result.append((fmt_pre + datatype_fmt, field.offset, field.name))

        result.sort(key=(lambda tup: tup[2]))
        return result

    def get_ndarray_from_msg(self, msg, topic, frame_idx):
        """
        Get point cloud data from bag message using tools in ros
        """
        fields = self.get_struct_fmt_map(msg.is_bigendian, msg.fields)
        arrays = []
        frame_col_name = 'frame_idx'
        if frame_idx != None:
            fields = [(None, None, frame_col_name)] + fields
        num_points = msg.row_step / msg.point_step
        if topic == '/rslidar_points':
            num_points = 78750
        for f in fields:
            if f[2] != frame_col_name:
                if num_points > 0:
                    arrays.append(np.ndarray(int(num_points), f[0], msg.data, f[1], msg.point_step))
                else:
                    arrays.append(np.ndarray([0]))
            else:
                arrays.append(np.full(int(num_points), frame_idx, dtype=np.int32))

        arrays2 = np.swapaxes(arrays, 0, 1)
        return num_points, [f[2] for f in fields], arrays2

    def get_bag_data(self, file_path, topic, FrameLimit):
        """
        Get point cloud data from bag
        """
        self.frame_count = 0
        array0 = np.array(0)
        points = []
        info = rosbag.Bag(file_path).get_message_count(topic)
        try:
            if info == 0 and topic == '/iv_points':
                topic = 'iv_points'
            bag_data = rosbag.Bag(file_path).read_messages(topic)
            if topic == '/rslidar_points':
                for topic, msg, t in bag_data:
                    arrays = list(point_cloud2.read_points(msg))
                    for i in range(len(arrays)):
                        points.append(list(arrays[i]) + [self.frame_count])

                    while self.frame_count == 0:
                        fields = msg.fields
                        array0 = np.zeros(len(fields) + 1)
                        break

                    self.frame_count += 1
                    array0 = np.vstack((array0, np.array(points)))
                    points.clear()
                    if self.frame_count == FrameLimit[1] + 2:
                        array0 = array0[1:]
                        fields = ['x', 'y', 'z', 'intensity', 'frame_idx']
                        return array0, fields

            else:
                for topic, msg, t in bag_data:
                    pt_num, fields1, arrays = self.get_ndarray_from_msg(msg, topic, self.frame_count)
                    while self.frame_count == 0:
                        fields = fields1
                        array0 = np.zeros(len(fields))
                        break

                    self.frame_count += 1
                    array0 = np.vstack((array0, arrays))
                    if self.frame_count == FrameLimit[1] + 2:
                        array0 = array0[1:]
                        return array0, fields

                array0 = array0[1:]
                return array0, fields
        except Exception as e:
            try:
                print(e)
            finally:
                e = None
                del e

    def get_pcd_data(self, file_path):
        """
        Get point cloud data from pcd
        """
        pts = []
        data = open(file_path, mode='r').readlines()
        pts_num = eval(data[8].split(' ')[-1])
        fields = data[1].strip('\nFIELDS ').split(' ')
        for line in data[10:]:
            p = line.strip('\n').split(' ')
            pts.append(p)

        assert len(pts) == pts_num
        res = np.zeros((pts_num, len(pts[0])), dtype=float)
        for i in range(pts_num):
            res[i] = pts[i]

        return res, fields

    def get_pcap_data(self, file_path):
        """
        Get point cloud data from pcap
        """
        dir = os.path.dirname(file_path)
        for filename in os.listdir(dir):
            if '.pcd' in filename:
                print(dir + '/' + filename)
                arrays, fields = self.get_pcd_data(dir + '/' + filename)
                array = np.vstack((array, arrays))

        return array, fields

    def get_file_data(self, file_path, topic, FrameLimit):
        """
        Get point cloud data from file; four file formats are currently supported
        """
        fields = ''
        if '.pcd' in file_path:
            res, fields = self.get_pcd_data(file_path)
        elif '.csv' in file_path:
            res = pd.read_csv(file_path)
            fields = list(res.columns.values)
            res = res.values
        elif '.bag' in file_path:
            res, fields = self.get_bag_data(file_path, topic, FrameLimit)
        elif '.pcap' in file_path:
            res, fields = self.get_pcap_data(file_path)
        else:
            print('Get data from file failed')

        # Make the data order conform to 'fields_wanted'
        for i in range(len(self.fields_wanted)):
            for j in range(len(fields)):
                if self.fields_wanted[i] == fields[j]:
                    self.index_sort[i] = j
                    break
                else:
                    self.index_sort[i] = -1
        j = 0
        new_sort = np.zeros(len(fields), dtype=int)
        for i in range(len(self.index_sort)):
            if self.index_sort[i] != -1:
                new_sort[j] = self.index_sort[i]
                j += 1
        sorted_fields = list(range(len(new_sort)))
        for i in range(len(new_sort)):
            sorted_fields[i] = fields[new_sort[i]]
        res = res[:, new_sort]
        print(sorted_fields)
        
        # Differentiating LiDAR types through scanning beam patterns
        max_scanline = max(res[:, self.scanline])
        if max_scanline <= 39:
            self.LiDAR_model = self.LiDAR_model[0]
        elif max_scanline > 39 and max_scanline <= 127:
            self.LiDAR_model = self.LiDAR_model[2]
        elif max_scanline > 127:
            self.LiDAR_model = self.LiDAR_model[1]
            
        return res, sorted_fields

    def get_fold_files(self, path):
        """
        Determine whether it is a calibration bag file.
        """
        count = 0
        result = []
        count1 = 0
        result1 = []
        for root, dirs, files in os.walk(path):
            for filename in sorted(files):
                if 'Cali' in filename:
                    if '.bag' in filename:
                        result.append(filename)
                        count += 1
                if '.bag' in filename:
                    result1.append(filename)
                    count1 += 1

        if count:
            print('find %d Calibration bag files' % count)
            print('using', ''.join(result[-1]))
        if count1:
            print('find %d normal bag files' % count1)
            print('using', ''.join(result1[-1]))
        if count1:
            return result1
        return None, None


class Analysis:

    def __init__(self):
        self.extract = Extract()
        self.fitting_plane = Fitting_plane()
        self.q = Queue()

    def point_number_of_FOV_per_frame(self, arrays):
        x = range(len(arrays))
        y = arrays
        plt.plot(x, y)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('point number of FOV')
        plt.show()

    def point_number_of_target_per_frame(self, arrays):
        x = range(len(arrays))
        y = arrays
        plt.plot(x, y)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('point number of target')
        plt.show()

    def mean_intensity_per_frame(self, arrays):
        x = range(len(arrays))
        y = arrays
        plt.plot(x, y)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('intensity of target')
        plt.show()

    def point_of_target_show(self, arrays):
        z = arrays[:, self.extract.x]
        y = arrays[:, self.extract.y]
        x = arrays[:, self.extract.z]
        ax = plt.axes(projection='3d')
        ax.scatter3D(x, y, z, cmap='b', c='b')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_zlim((min(z) - 0.2, max(z) + 0.2))
        plt.title('point extract')
        plt.xlim((min(x) - 0.2, max(x) + 0.2))
        plt.show()

    def get_filepath_distance(self, file_path):
        pattern = r'(\d+)m'
        match = re.search(pattern, file_path)
        if match:
            Theory_distance = float(match.group(1))
            return Theory_distance
        else:
            print('File path No match distance')
            return 0
        
    def get_filepath_ref(self, file_path):
        pattern = r'(\d+)ref'
        match = re.search(pattern, file_path)
        if match:
            Theory_ref = float(match.group(1))
            return Theory_ref
        else:
            print('File path No match ref')
            return 0

    def Calculate_data(self, file_path, FrameLimit, BoundingBox, IntensityBox, case, topic):
        """
        Calculate the POD, average number of points per frame in FOV/Bounding, and reflectivity information(Main)
        """
        results = []
        FOVROI = ['NoResult']
        POD = ['NoResult']
        PointsNum = ['NoResult']
        MeanItensity = ['NoResult']
        Precision = ['NoResult']
        self.extract.topic = self.extract.topics[topic]
        Theory_distance = self.get_filepath_distance(file_path)
        Theory_ref = self.get_filepath_ref(file_path)
        
        if len(BoundingBox) == 0:
            BoundingBox = [-3, 3, -3, 3, Theory_distance - 0.3, Theory_distance + 0.3]

        if len(IntensityBox) == 0:
            if Theory_ref == 10:
                IntensityBox = [0, 45]
            elif Theory_ref == 40:
                IntensityBox = [20, 80]
            elif Theory_ref == 80:
                IntensityBox = [40, 110]
            else:
                print('No match ref')
        
        if self.extract.topic != '/iv_points' and self.extract.topic != 'iv_points':
            self.extract.x = 0
            self.extract.y = 1
            self.extract.z = 2
            self.extract.intensity = 3
            self.extract.f = 4
        pts_arrays, fields = self.extract.get_file_data(file_path, self.extract.topic, FrameLimit)
        pts_arrays = pts_arrays[~np.isnan(pts_arrays).any(axis=1)]
        pts_sel = self.Filter_xyz(pts_arrays, FrameLimit, BoundingBox, IntensityBox)
        pointnum_perframe = np.bincount(pts_sel[:, self.extract.f].astype(int).flatten(), minlength=FrameLimit[1]+1)
        pointnum_perframe = pointnum_perframe[FrameLimit[0]:]
        frame_counts = len(pointnum_perframe)
        print('共分析： ', frame_counts, '帧')
        if case[0] == 1:
            FOVROI = self.Analyze_FOVROI_Angular_Resolution(pts_arrays, fields)
        if case[1] == 1:
            MeanItensity = self.Meanintensity_perframe(pts_sel, self.extract.f, self.extract.intensity, FrameLimit)
        if case[2] == 1:
            PointsNum = self.Calculate_points_num(pts_sel, pts_arrays, pointnum_perframe, frame_counts, FrameLimit)
        if case[3] == 1:
            i = FrameLimit[0]
            distance = self.get_points_distance(pts_sel)
            pts_sel_temp = pts_sel[np.where((pts_sel[:, self.extract.f] >= i) & (pts_sel[:, self.extract.f] < i + distance / 3))]
            Precision = self.fitting_plane.Extract_point_fitting_plane(pts_sel_temp, FrameLimit, self.extract.topic)
        if case[4] == 1:
            POD = self.POD(pts_sel, frame_counts, len(pts_sel[:, 4]) / frame_counts, BoundingBox)
        results.extend([FOVROI])
        results.extend([MeanItensity])
        results.extend(PointsNum)
        results.extend([Precision])
        results.extend(POD)
        ExcelOperation.WritetToExcel(results, file_path)
        return results

    def get_points_distance(self, pts_sel):
        """
        Calculate_points_average_distance
        """
        distance = 0
        x = pts_sel[:, self.extract.x]
        y = pts_sel[:, self.extract.y]
        z = pts_sel[:, self.extract.z]
        for i in range(len(pts_sel[:, 4])):
            distance = distance + math.sqrt(x[i] ** 2 + y[i] ** 2 + z[i] ** 2)

        distance = distance / len(pts_sel[:, 4])
        print('distance=:', distance)
        return distance

    def Calculate_points_num(self, pts_sel, pts_arrays, pointnum_perframe, frame_count, FrameLimit):
        """
        Calculate_data_points
        """
        pts_arrays = pts_arrays[np.where((pts_arrays[:, self.extract.f] > FrameLimit[0] - 1) & (pts_arrays[:, self.extract.f] < FrameLimit[1] + 1))]
        fov_pointnum_perframe = np.bincount(pts_arrays[:, self.extract.f].astype(int).flatten())[FrameLimit[0]:]
        target_mean_points = len(pts_sel) / frame_count
        pts_sel_var = np.var(pointnum_perframe)
        pts_sel_std_1 = np.std(pointnum_perframe)
        pts_sel_std_2 = np.std(pointnum_perframe, ddof=1)
        max_width_height = self.Get_Max_Width_Height(pts_sel)
        print('target mean points:', target_mean_points)
        print('target points per frame:', pointnum_perframe)
        print('目标点方差为：%f' % pts_sel_var)
        print('目标点总体标准差为: %f' % pts_sel_std_1)
        print('目标点样本标准差为: %f' % pts_sel_std_2)
        results = [
         ['FOV mean points', pts_arrays.shape[0] / frame_count, 'FOV sum points', pts_arrays.shape[0],
          'FOV pointnum perframe', fov_pointnum_perframe.tolist(), 'target mean points', target_mean_points,
          'target points Variance', pts_sel_var, 'target points Standard Deviation', pts_sel_std_1,
          'target points per frame', pointnum_perframe.tolist(), 'Max Width', max_width_height[0],
          'Max Width', max_width_height[1]]]
        return results

    def Meanintensity_perframe(self, pts_sel, f, inten, FrameLimit):
        """
        Calculate_points_intensity
        """
        meanintensity_perframe = []
        frame_min = FrameLimit[0]
        frame_max = FrameLimit[1]
        for i in range(frame_min, frame_max):
            a = pts_sel[np.where(pts_sel[:, f] == i)]
            meanintensity_perframe.append(np.mean(a[:, inten]))
            intensity = np.mean(pts_sel[:, inten])

        pts_sel_var = np.var(meanintensity_perframe)
        pts_sel_std_1 = np.std(meanintensity_perframe)
        pts_sel_std_2 = np.std(meanintensity_perframe, ddof=1)
        print('mean intensity per frame:', meanintensity_perframe)
        print('mean intensity:', intensity)
        print('intensity方差为: %f' % pts_sel_var)
        print('intensity总体标准差为: %f' % pts_sel_std_1)
        print('intensity样本标准差为: %f' % pts_sel_std_2)
        meanintensity_perframe.extend([intensity])
        results = ['Mean Intensity', intensity, 'Mean Intensity Per Frame', meanintensity_perframe,
         'Intensity Variance', pts_sel_var, 'Intensity Standard Deviation', pts_sel_std_1]
        return results

    def Filter_xyz(self, input_array, framelimit, bounding_box, intensity_bounding):
        """
        Select points within the ‘bounding_box’
        """
        x = self.extract.x
        y = self.extract.y
        z = self.extract.z
        f = self.extract.f
        # if len(input_array) < 1:
        #     input_array
        if bool(framelimit):
            frame_max = framelimit[1]
            frame_min = framelimit[0]
            input_array = input_array[np.where((input_array[:, f] > frame_min - 1) & (input_array[:, f] < frame_max + 1))]
        if bool(bounding_box):
            xmin, xmax, ymin, ymax, zmin, zmax = (
             bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3],
             bounding_box[4], bounding_box[5])
            input_array = input_array[np.where((input_array[:, x] > xmin) & (input_array[:, x] < xmax))]
            input_array = input_array[np.where((input_array[:, y] > ymin) & (input_array[:, y] < ymax))]
            input_array = input_array[np.where((input_array[:, z] > zmin) & (input_array[:, z] < zmax))]
        if bool(intensity_bounding):
            input_array = input_array[np.where((input_array[:, 6] >= intensity_bounding[0]) & (input_array[:, 6] <= intensity_bounding[1]))]
        return input_array

    def POD(self, pts, frame_count, real_points, bounding):
        """
        Calculate points POD within the ‘bounding_box’
        """
        distance = self.get_points_distance(pts)
        
        index = 0
        if self.extract.LiDAR_model == 'W':
            index = 1
        elif self.extract.LiDAR_model == 'E':
            index = 2
            
        Horizontal_R = [0.09, 0.13, 0.086] # K, W, E
        Vertical_R = [0.08, 0.37, 0.182] 
        V_Int = math.radians(Horizontal_R[index]) * distance
        R_Int = math.radians(Vertical_R[index]) * distance
        eps = max(V_Int, R_Int)
        pts1 = Cluster.find_largest_cluster_DBSCAN(pts, 1.5 * eps, 50)
        pts = pts[np.where((pts[:, 3] >= max(pts1[:,1]) - 1.55) & (pts[:, 3] <= max(pts1[:,1])))]
        pts1 = Cluster.find_largest_cluster_DBSCAN(pts, eps + 0.01, 50)
        pts = self.Filter_xyz(pts, None, [min(pts1[:,0]), max(pts1[:,0]), min(pts1[:,1]), max(pts1[:,1]), min(pts1[:,2]), max(pts1[:,2])], None)
        res = self.Get_Max_Width_Height(pts)
        Width = res[0]
        Height = res[1]
        # LaserSpot = round(0.00087 * distance * 2, 3)
        # Height = bounding[1] - bounding[0] - LaserSpot * 1.2
        # Width = bounding[3] - bounding[2] - LaserSpot * 1.35
        if distance >= 1000:
            number = 0
            ideal_points = system_pod_points[len(system_pod_points) - 1]
            for i in range(len(system_pod_points) - 1):
                row = system_pod_points[i][0]
                colmin = system_pod_points[i][1]
                colmax = system_pod_points[i][2]
                pts1 = pts[np.where((pts[:, 1] == row) & (pts[:, 2] >= colmin) & (pts[:, 2] <= colmax))]
                number += len(pts1[:, 1])

            real_points = number / frame_count
        else:
            row = math.atan(Width / (2 * distance)) * 2 * 180 / math.pi / Horizontal_R[index] + 1
            col = math.atan(Height / (2 * distance)) * 2 * 180 / math.pi / Vertical_R[index] + 1
            ideal_points = row * col
            pod = '{:.2%}'.format(real_points / ideal_points)
        if ideal_points < real_points:
            pod = '{:.2%}'.format(1 + (real_points - ideal_points) / (ideal_points * 10))
        print('idea_points', ideal_points)
        print('real_points', real_points)
        print('pod:', pod)
        results = [['distance', '{:.2f}'.format(distance), 'ideal_points', '{:.2f}'.format(ideal_points), 'real_points', '{:.2f}'.format(real_points), 'POD', '{:.2%}'.format(real_points / ideal_points)]]
        return results

    def Analyze_ROI_MAXClearance(self, pts, fields):
        """
        Analyze ROI MAXClearance
        """
        pts = pts[np.where((pts[:, 2] > 740) & (pts[:, 13] == 3))]
        Scanline_max = int(max(pts[:, 1]))
        Scanline_min = int(min(pts[:, 1]))
        Channel_max = int(max(pts[:, 12]))
        Channel_min = int(min(pts[:, 12]))
        x = np.zeros([Scanline_max - Scanline_min + 1, Channel_max - Channel_min + 1])
        maxClerance = x
        for i in range(Scanline_min, Scanline_max + 1):
            for j in range(Channel_min, Channel_max + 1):
                x[(i - Scanline_min, j)] = np.mean(pts[(np.where((pts[:, 1] == i) & (pts[:, 12] == j)), 3)])

        for j in range(Channel_min, Channel_max):
            for i in range(len(x[:, 0])):
                maxClerance[(i, j)] = x[(i, j + 1)] - x[(i, j)]

        print('ROIMaxClerance:', max(map(max, maxClerance)))

    def Analyze_FOVROI_Angular_Resolution(self, pts, fields):
        """
        Analyze ROI&Non-ROI Angular Resolution
        """
        if 'flags' in fields:
            pts = pts[np.where((pts[:, 7] > 10) & (pts[:, 7] < 12))]
            temp = np.zeros(len(fields) + 2)
        result = self.q.get()
        if 'flags' in fields:
            pts = pts[np.where((pts[:, 7] > 10) & (pts[:, 7] < 12))]
            temp = np.zeros(len(fields) + 2)
            if self.extract.LiDAR_model == 'K': # I/G/K/K24
                for i in range(len(pts[:, 0])):
                    channel = np.append((pts[i]), [int('{:08b}'.format(int(pts[(i, 0)]))[-2:], 2)], axis=0)
                    roi = np.append(channel, [int('{:08b}'.format(int(pts[(i, 0)]))[-3:-2], 2)], axis=0)
                    temp = np.vstack([temp, roi])
                jobs = []
                for i in range(2):
                    p = Process(target=(self.Calculate_Angle_Resolution), args=(temp, i, self.q))
                    jobs.append(p)
                    p.start()

                for p in jobs:
                    p.join()

                result = [self.q.get() for j in jobs]
            
            elif self.extract.LiDAR_model == 'E':
                self.Calculate_Angle_Resolution(pts, -1, self.q)
                result = self.q.get()

            elif self.extract.LiDAR_model == 'W':
                self.Calculate_Angle_Resolution(pts, -2, self.q)
                result = self.q.get()
        print(result)
        return result

    def Calculate_Angle_Resolution(self, temp, I, q):
        """
        Calculate ROI&Non-ROI Angular Resolution
        """
        roi_line = 56
        scanline_coef = 4
        widest_line = temp[np.where(temp[:, 1] == 64)] #Robin E
        # widest_line = temp[np.where(temp[:, 1] == 0)] #Robin E

        if I >= 0: # Falcon I/G/K/K24
            temp = temp[np.where(temp[:, -1] == I)]
            temp1 = temp[np.where((temp[:, 4] > -0.04) & (temp[:, 4] < 0.04))]
            top_scanline_id = temp1[np.where(temp1[:, 3] == max(temp1[:, 3]))][:, 1][0]
            top_channel_id = temp1[np.where(temp1[:, 3] == max(temp1[:, 3]))][:, -2][0]
            bottom_scanline_id = temp1[np.where(temp1[:, 3] == min(temp1[:, 3]))][:, 1][0]
            bottom_channel_id = temp1[np.where(temp1[:, 3] == min(temp1[:, 3]))][:, -2][0]
            bottom_line = temp[np.where((temp[:, 1] == bottom_scanline_id) & (temp[:, -2] == bottom_channel_id))]
            top_line = temp[np.where((temp[:, 1] == top_scanline_id) & (temp[:, -2] == top_channel_id))]
            widest_line = bottom_line
        
        if I <= 0: #Robin W/E
            roi_line = 0
            scanline_coef = 1
            bottom_line = temp[np.where(temp[:, 1] == min(temp[:, 1]))]
            top_line = temp[np.where(temp[:, 1] == max(temp[:, 1]))]
            if I == -2:
                widest_line = temp[np.where(temp[:, 1] == max(temp[:, 1]) - 1)] # Robin W
            
        widest_line_right = widest_line[np.where(widest_line[:, 4] == max(widest_line[:, 4]))]
        widest_line_left = widest_line[np.where(widest_line[:, 4] == min(widest_line[:, 4]))]
        bottom_point = bottom_line[np.where((bottom_line[:, 4] > -0.05) & (bottom_line[:, 4] < 0.05))]
        bottom_point = bottom_point[np.where(bottom_point[:, 3] == min(bottom_point[:, 3]))]
        top_point = temp[np.where((temp[:, 4] > -0.05) & (temp[:, 4] < 0.05))]
        top_point = top_point[np.where(top_point[:, 3] == max(top_point[:, 3]))]

        # line_num = 128
        line_num = (max(temp[:, 1]) - min(temp[:, 1]) + 1) * scanline_coef - roi_line
        points_num = max(widest_line[:, 2]) - min(widest_line[:, 2]) + 1
        Bx = bottom_point[(0, 3)]
        Tx = top_point[(0, 3)]
        Ly = widest_line_left[(0, 4)]
        Ry = widest_line_right[(0, 4)]
        Lz = widest_line_left[(0, 5)]
        Rz = widest_line_right[(0, 5)]
        Bz = bottom_point[(0, 5)]
        Tz = top_point[(0, 5)]
        Hangle = np.degrees(np.arccos((Ly * Ry + Lz * Rz) / np.sqrt((pow(Ly, 2) + pow(Lz, 2)) * (pow(Ry, 2) + pow(Rz, 2)))))
        Vangle = np.degrees(np.arccos((Bx * Tx + Bz * Tz) / np.sqrt((pow(Bx, 2) + pow(Bz, 2)) * (pow(Tx, 2) + pow(Tz, 2)))))
        H_Resolution = Hangle / points_num
        V_Resolution = Vangle / line_num
        print('Hangle:', Hangle, 'H_Resolution:', H_Resolution, 'Vangle:', Vangle, 'V_Resolution:', V_Resolution, 'ROI:', I)
        q.put(['Hangle:', Hangle, 'H_Resolution:', H_Resolution, 'Vangle:', Vangle, 'V_Resolution:', V_Resolution, 'ROI:', I])
        # return Hangle, H_Resolution, Vangle, V_Resolution

    def Calculate_Center_Of_Mess(self, pts_sel):
        """
        Calculate points center of mess
        """
        print('Calculating the center of mass x y z:', np.mean((pts_sel[:, 3:6]), axis=0))

    def Get_Max_Width_Height(self, pts_sel):
        """
        Calculate target points Max Width&Height
        """
        max_width = max(pts_sel[:, self.extract.y]) - min(pts_sel[:, self.extract.y])
        max_height = max(pts_sel[:, self.extract.x]) - min(pts_sel[:, self.extract.x])
        results = [max_width, max_height]
        return results
    
class Fitting_plane:
    
    def __init__(self):
        pass

    def fitting_plane_LSM(self, xyzs):
        """
        使用最小二乘法拟合平面

        Args:
            xyzs (ndarray): 包含

        Returns:
            _type_: _description_
        """
        x, y, z = xyzs.T
        A = np.column_stack((x, y, np.ones_like(x)))
        B = z.reshape(-1, 1)

        # 使用numpy.linalg.lstsq()计算最小二乘解
        X, residuals, _, _ = np.linalg.lstsq(A, B, rcond=None)

        # 提取拟合结果的系数
        a, b, c = X.flatten()

        # 打印拟合结果
        # print('拟合结果：z = %.3f * x + %.3f * y + %.3f' % (a, b, c))
        residual_squared_sum = np.sum(residuals)
        standard_deviation = np.sqrt(residual_squared_sum / len(x)) * 100

        # print('标准差：%.3f' % standard_deviation)

        return a, b, c, standard_deviation
    
    def fit_plane_Ransac(self, xyzs, threshold=0.01, max_iterations=1000):
        """
        使用RANSAC算法从三维点云中拟合最佳平面
        参数:
            - xyzs: 三维点云数组，形状为 (n, 3)。每一行代表一个点的 x、y 和 z 坐标。
            - threshold: 用于判断点是否属于拟合平面的阈值，默认为 0.01。
            - max_iterations: RANSAC算法的最大迭代次数，默认为 1000。
        返回:
            - best_model: 拟合的平面模型，形式为 (a, b, c) 其中平面方程为 z = ax + by + c。
            - best_std_dev: 最佳平面模型下的最小二乘拟合平面标准差结果。
        """
        best_model = None
        best_num_inliers = 0
        best_std_dev = float('inf')
        for i in range(max_iterations):
            # 随机选择三个点作为样本
            sample_indices = np.random.choice(len(xyzs), 3, replace=False)
            sample_xyzs = xyzs[sample_indices]

            # 使用这三个点拟合平面
            model = self.fit_plane(sample_xyzs)

            # 计算其他所有点到该平面的距离，并用最小二乘拟合内点平面求标准差
            dists = self.pts_to_plane_distance(model, xyzs)
            inlier_indices = np.where(dists < threshold)[0]
            num_inliers = len(inlier_indices)
            inliers_temp = xyzs[inlier_indices]
            a, b, c, std_dev = self.fitting_plane_LSM(inliers_temp)

            # 更新最佳模型
            if num_inliers > best_num_inliers or (num_inliers == best_num_inliers and std_dev < best_std_dev):
                best_std_dev = std_dev
                best_model = a, b, c
                best_num_inliers = num_inliers
        print('best_model:', best_model, 'best_num_inliers', best_num_inliers, 'best_std_dev', best_std_dev)
        return best_model, best_std_dev

    def fit_plane(self, xyzs):
        """
        使用向量计算平面方程
        参数:
            - points: 三维点云数组，形状为 (3, 3)。每一行代表一个点的 x、y 和 z 坐标。
        返回:
            - model: 拟合的平面模型，形式为 (a, b, c, d) 其中平面方程为 ax + by + cz + d = 0。
        """
        P1 = xyzs[0]
        P2 = xyzs[1]
        P3 = xyzs[2]
        v1 = P2 - P1
        v2 = P3 - P1

        # 计算法向量 N
        N = np.cross(v1, v2)

        # 使用法向量 N 和已知点 P1 得到平面方程
        A, B, C = N
        D = -(A * P1[0] + B * P1[1] + C * P1[2])
        model = A, B, C, D
        return model
    
    def pts_to_plane_distance(self, model, xyzs):
        distance = []
        A, B, C, D = model
        x, y, z = xyzs.T
        denominator = math.sqrt(A ** 2 + B ** 2 + C ** 2)
        for i in range(x.shape[0]):
            numerator = abs(A * x[i] + B * y[i] + C * z[i] + D)
            distance.append(numerator / denominator)
        distance = np.nan_to_num(distance, nan=0.0, posinf=0.0, neginf=0.0)
        return distance
    
    def Extract_point_fitting_plane(self, pts_sel, FrameLimit, topic):
        """"""
        i = FrameLimit[0]
        max_iterations = 1500
        n = pts_sel.shape[0]
        goal_inliers = n * 0.95
        threshold= 0.04
        best_std_dev = float('inf')
        # points data
        xyzs = pts_sel[:,3:6]
        if topic != '/iv_points' and topic != 'iv_points':
            xyzs = pts_sel[:,0:3]

        # RANSAC
        for i in range(5):
            model, std_dev = self.fit_plane_Ransac(xyzs, threshold, max_iterations)
            if std_dev < best_std_dev:
                best_std_dev = std_dev
                best_model = model
        a, b, c = best_model
        Precision = ['a', a, 'b', b, 'c', c, 'sigma', best_std_dev]
        print(Precision)
        return Precision
    
class Cluster:
        
    def __init__(self):
        pass
    
    def find_largest_cluster_KMeans(points):
        # 使用KMeans进行粗略聚类
        kmeans = KMeans(n_clusters=150)  # 或者使用 DBSCAN(n_clusters=10)
        labels = kmeans.fit_predict(points[:, 3:6])
        
        # 找出每个类别的点数
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # 找到点数最多的类别
        largest_cluster_label = unique_labels[np.argmax(counts)]
        
        # 输出点数最多的类别
        largest_cluster = points[:, 3:6][labels == largest_cluster_label]
        return largest_cluster
    
    def find_largest_cluster_DBSCAN(points, e, min_samples):
        # 使用DBSCAN进行粗略聚类
        print('eps:', e, 'min_samples:', min_samples)
        dbscan = DBSCAN(eps=e, min_samples=min_samples)  # 调整eps和min_samples参数
        labels = dbscan.fit_predict(points[:, 3:6])
        
        # 找出每个类别的点数
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # 找到点数最多的类别
        largest_cluster_label = unique_labels[np.argmax(counts)]
        
        # 输出点数最多的类别
        largest_cluster = points[:, 3:6][labels == largest_cluster_label]
        return largest_cluster
    
    def find_largest_cluster_HDBSCAN(points, e, min_samples):
        # 使用HDBSCAN进行聚类
        clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=min_samples)
        cluster_labels = clusterer.fit_predict(points[:, 3:6])

        # 绘制聚类结果
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for cluster in np.unique(cluster_labels):
            if cluster == -1:
                color = 'gray'  # 噪声点
            else:
                color = plt.cm.jet(cluster / np.max(cluster_labels))  # 根据簇的标签选择颜色
            ax.scatter(points[cluster_labels == cluster][:, 3],
                        points[cluster_labels == cluster][:, 4],
                        points[cluster_labels == cluster][:, 5],
                        c=color, marker='o')

        plt.show()


if __name__ == '__main__':
    system_pod_points = [[22, 634, 635], [23, 635, 636], 4]
    intensity_bounding = []
    file_path = [
    "/home/demo/Desktop/004TestData/0815_robinw/10ref10m_2023-08-15-14-39-18.bag",
    "/home/demo/Desktop/004TestData/0815_robinw/10ref20m_2023-08-15-14-37-14.bag",
    "/home/demo/Desktop/004TestData/0815_robinw/10ref30m_2023-08-15-14-35-51.bag",
    "/home/demo/Desktop/004TestData/0815_robinw/10ref40m_2023-08-15-14-34-37.bag",
    "/home/demo/Desktop/004TestData/0815_robinw/10ref50m_2023-08-15-14-33-05.bag",
    "/home/demo/Desktop/004TestData/0815_robinw/10ref60m_2023-08-15-14-13-04.bag",
    "/home/demo/Desktop/004TestData/0815_robinw/10ref70m_2023-08-15-14-07-01.bag"]
    bounding00 = [-0.55, 1.3, 0.1, 1.75, 9.9, 10.35] #10%ref
    bounding0 = [-0.75, 1.1, 0.7, 2.35, 19.9, 20.35]
    bounding1 = [-0.7, 1.15, 1.5, 3.1, 29.9, 30.4]
    bounding2 = [-0.75, 1.1, 2.5, 4.15, 39.9, 40.45]
    bounding3 = [-0.9, 0.9, 3.5, 5.05, 49.5, 50.15]
    bounding4 = [-0.6, 1.2, 2, 3.5, 59.7, 60.3]
    bounding5 = [-1.05, 0.85, 1.45, 2.95, 69.9, 70.4]
    bounding6 = [0.75, 2.25, -2.2, -0.7, 139.95, 140.2]
    bounding7 = [0.9, 2.55, -2.65, -0.9, 160, 160.2]
    bounding8 = [0.9, 2.45, -1.1, 0.55, 179.95, 180.3]
    bounding9 = [1, 2.65, -0.75, 0.95, 200.2, 200.6]
    bounding10 = [1.15, 2.75, -0.2, 1.35, 219.9, 220.15]
    bounding11 = [1.2, 2.8, 0.1, 1.65, 240, 240.35]
    bounding12 = [1.45, 2.95, -0.5, 1.25, 249.9, 250.3]
    bounding13 = [-0.3, 0.5, -0.35, 0.5, 2.05, 2.25] #40%ref
    bounding14 = [-0.1, 0.7, -0.65, 0.15, 20, 20.1]
    bounding15 = [-0.05, 0.85, -0.45, 0.45, 40.5, 40.7]
    bounding01 = [0, 0.95, -0.85, 0.05, 60.1, 60.3]
    bounding16 = [0, 1, -3.9, -3.0, 79.7, 80]
    bounding17 = [0.2, 1, -3.4, -2.55, 99.9, 100.1]
    bounding18 = [-0.3, 0.5, -0.35, 0.5, 2.1, 2.3] #90%ref
    bounding19 = [-0.1, 0.7, -0.25, 0.65, 19.9, 20.1]
    bounding20 = [-0.05, 0.85, -0.85, 0.05, 40.1, 40.2]
    bounding21 = [0, 0.95, -1.3, -0.4, 60.2, 60.3]
    bounding22 = [0.08, 0.98, -2.75, -1.8, 80, 80.1]
    bounding23 = [0.05, 1, -3.4, -2.4, 99.9, 100.1]

    bounding = [bounding00, bounding0, bounding1, bounding2, bounding3, bounding4,
                bounding5, bounding6, bounding7, bounding8, bounding9,
                bounding10, bounding11, bounding12,
                bounding13, bounding14,
                bounding15, bounding01, bounding16, bounding17,
                bounding18, bounding19, bounding20, bounding21, bounding22,
                bounding23]

    
    # for i in range(len(file_path)):
    #     Analysis().Calculate_data(file_path[i], [0, 100], bounding[i], [], [0, 1, 1, 1, 1], 0)
    Analysis().Calculate_data('/home/demo/Desktop/004TestData/20418_10ref60m_2023-02-27-10-38-17.bag', [0, 100], [], [], [0, 0, 0, 0, 1], 0)
    print('What have done is done')