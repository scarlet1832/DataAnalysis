import math
import os
from multiprocessing import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rosbag
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
import random
from ransac import *
import matplotlib
from scipy.spatial import distance
from scipy import stats

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

        for i in range(len(self.fields_wanted)):  # Make the data order conform to 'fields_wanted'
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
        self.q = Queue()

    def augment(self, xyzs):
        axyz = np.ones((len(xyzs), 4))
        axyz[:, :3] = xyzs
        return axyz

    def estimate(self, xyzs):
        axyz = self.augment(xyzs[:3])
        return np.linalg.svd(axyz)[-1][-1, :]
    
    def is_inlier(self, coeffs, xyz, threshold):
        return np.abs(coeffs.dot(self.augment([xyz]).T)) < threshold

    def plot_plane(self, a, b, c, d):
        xx, yy = np.mgrid[:10, :10]
        return xx, yy, (-d - a * xx - b * yy) / c
    
    def calculate_std_dev(self, coeffs, data):
        distances = []
        for xyz in data:
            dist = np.abs(coeffs.dot(self.augment([xyz]).T))
            distances.append(dist)
        return np.std(distances)
    
    def extract_point_fitting_plane_2(self, data, sample_size, goal_inliers, max_iterations, stop_at_goal=True, random_seed=None):
        best_ic = 0
        best_model = None
        best_std_dev = float('inf')
        random.seed(random_seed)
        data = list(data)
        inliers = [] # 存储内点
        outliers = [] # 存储离群点
        inliers_number = []
        outliers_number = []
        best_std_dev_number = []
        for i in range(max_iterations):
            s = random.sample(data, int(sample_size))
            m = self.estimate(s)
            ic = 0
            current_inliers = [] # 存储当前内点
            curren_outliers = [] # 存储当前外点
            for j in range(len(data)):
                if self.is_inlier(m, data[j], 0.005): # 修改调用 is_inlier 函数的方式，传入阈值
                    ic += 1
                    current_inliers.append(data[j])
                else:
                    curren_outliers.append(data[j])
            std_dev = self.calculate_std_dev(m, current_inliers)
            if ic > best_ic or (ic == best_ic and std_dev < best_std_dev):
                best_ic = ic
                best_model = m
                best_std_dev = std_dev
                inliers = current_inliers # 更新内点列表
                outliers = curren_outliers # 更新外点列表
                inliers_number.append(len(inliers))
                outliers_number.append(len(outliers))
                best_std_dev_number.append(best_std_dev)
                # if ic > goal_inliers and stop_at_goal:
                #     break

        x = list(range(len(inliers_number)))
        # 绘制三个列表的变化
        plt.plot(x, inliers_number, label='inliers_number')
        plt.plot(x, outliers_number, label='outliers_number')
        plt.plot(x, best_std_dev_number, label='best_std_dev_number')

        # 添加图例
        plt.legend()

        # 添加标题和标签
        plt.title('STD Variation')
        plt.xlabel('Index')
        plt.ylabel('Value')

        # 显示图形
        plt.show()
        print(s)
        print('estimate:', m,)
        print('# inliers:', ic)
        print('std deviation:', best_std_dev * 100)
        print('inliners number:', len(inliers), '\noutliners number:', len(outliers))
        print('took iterations:', i+1, 'best model:', best_model, 'explains:', best_ic)

        # 输出平面方程形式的拟合结果
        a, b, c, d = best_model
        # print('拟合结果：z = %.3f * x + %.3f * y + %.3f' % (-a, -b, -c))

        # 返回最佳模型、内点数和离群点数
        return a, b, c, best_std_dev

    def extract_point_fitting_plane_1(self, arrays):
        # if self.extract.topic != '/iv_points' and self.extract.topic != 'iv_points':
        #     z = arrays[:, 0]
        #     y = arrays[:, 1]
        #     x = arrays[:, 2]
        # else:
        #     x = arrays[:, 3]
        #     y = arrays[:, 4]
        #     z = arrays[:, 5]
        x, y, z = arrays.T
        
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
    
    def Extract_point_fitting_plane(self, pts_sel, FrameLimit):
        """
        extract_point_fitting_plane
        """
        i = FrameLimit[0]
        # distance = self.get_points_distance(pts_sel)
        # pts_sel = pts_sel[np.where((pts_sel[:, self.extract.f] >= i) & (pts_sel[:, self.extract.f] < i + distance / 2))]
        # a, b, y, sigma = self.extract_point_fitting_plane_1(pts_sel) 
        # Precision = ['a', a, 'b', b, 'y', y, 'sigma', sigma]
        n = pts_sel.shape[0]
        max_iterations = 1500
        goal_inliers = n * 0.95
        threshold= 0.04
        # test data
        xyzs = pts_sel[:,3:6]

        # RANSAC
        # a, b, c, d = self.extract_point_fitting_plane_2(xyzs, 3, goal_inliers, max_iterations)

        # x = pts_sel[:, 3]
        # y = pts_sel[:, 4]
        # z = pts_sel[:, 5]
        # ax = plt.axes(projection='3d')
        # ax.scatter3D(z, y, x, cmap='b', c='r')
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # plt.xlim((min(z) - 0.3, max(z) + 0.3))
        # plt.title('point extract')
        # plt.show()

        # a, b, c, d = m
        # xx, yy, zz = self.plot_plane(a, b, c, d)
        for i in range(10):
            self.fit_plane_ransac(xyzs, threshold, max_iterations)
        e, f, g = self.fit_plane_ransac(xyzs, threshold, max_iterations)
        return e
    
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
        # if case[0] == 1:
        #     FOVROI = self.Analyze_FOVROI_Angular_Resolution(pts_arrays, fields)
        # if case[1] == 1:
        #     MeanItensity = self.Meanintensity_perframe(pts_sel, self.extract.f, self.extract.intensity, FrameLimit)
        # if case[2] == 1:
        #     PointsNum = self.Calculate_points_num(pts_sel, pts_arrays, pointnum_perframe, frame_counts, FrameLimit)
        if case[3] == 1:
            Precision = self.Extract_point_fitting_plane(pts_sel, FrameLimit)
        # if case[4] == 1:
        #     POD = self.POD(pts_sel, frame_counts, len(pts_sel[:, 4]) / frame_counts, BoundingBox)
        results.extend([FOVROI])
        results.extend([MeanItensity])
        results.extend(PointsNum)
        results.extend([Precision])
        results.extend(POD)
        # ExcelOperation.WritetToExcel(results, file_path)
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
    ###########################################
    def fit_plane_ransac(self, points, threshold=0.01, max_iterations=1000):
        """
        使用RANSAC算法从三维点云中拟合最佳平面
        参数:
            - points: 三维点云数组，形状为 (n, 3)。每一行代表一个点的 x、y 和 z 坐标。
            - threshold: 用于判断点是否属于拟合平面的阈值，默认为 0.01。
            - max_iterations: RANSAC算法的最大迭代次数，默认为 1000。
        返回:
            - best_model: 拟合的平面模型，形式为 (a, b, c, d) 其中平面方程为 ax + by + cz + d = 0。
            - inliers: 内点，形状为 (m, 3)。m 为内点的数量。
            - outliers: 外点，形状为 (n-m, 3)。n 为总点数，m 为内点的数量。
        """
        best_model = None
        best_num_inliers = 0
        best_std_dev = float('inf')
        inliers = []
        outliers = []
        for i in range(max_iterations):
            # 随机选择三个点作为样本
            sample_indices = np.random.choice(len(points), 3, replace=False)
            sample_points = points[sample_indices]
            # 使用这三个点拟合平面
            model = self.fit_plane(sample_points)

            # 计算其他所有点到该平面的距离
            dists = self.pts_to_plane_distance(model, points)
            inlier_indices = np.where(dists < threshold)[0]
            num_inliers = len(inlier_indices)
            num_outliners = len(points) - num_inliers
            inlier_temp = points[inlier_indices]
            a, b, c, std_dev = self.extract_point_fitting_plane_1(inlier_temp)

            # 更新最佳模型
            if num_inliers > best_num_inliers or (num_inliers == best_num_inliers and std_dev < best_std_dev):
                best_std_dev = std_dev
                best_model = model
                best_num_inliers = num_inliers
                inliers = inlier_temp
                outliers = points[np.setdiff1d(np.arange(len(points)), inlier_indices)]
                # a, b, c, std_dev1 = self.extract_point_fitting_plane_1(inliers)
                # std_dev2 = np.std(dists) * 100
                # std_dev3 = self.calculate_std_dev(np.array(model), points.tolist()) * 100
                # print("标准差:", std_dev1, std_dev2, std_dev3, '\nbest_num_inliers', best_num_inliers)
        print('best_model:', best_model, 'best_num_inliers', best_num_inliers, 'best_std_dev', best_std_dev)
        return best_model, inliers, outliers

    def fit_plane(self, points):
        """
        使用向量计算平面方程
        参数:
            - points: 三维点云数组，形状为 (3, 3)。每一行代表一个点的 x、y 和 z 坐标。
        返回:
            - model: 拟合的平面模型，形式为 (a, b, c, d) 其中平面方程为 ax + by + cz + d = 0。
        """
        P1 = points[0]
        P2 = points[1]
        P3 = points[2]
        v1 = P2 - P1
        v2 = P3 - P1

        # 计算法向量 N
        N = np.cross(v1, v2)

        # 使用法向量 N 和已知点 P1 得到平面方程
        A, B, C = N
        D = -(A * P1[0] + B * P1[1] + C * P1[2])
        model = A, B, C, D
        return model
    
    def pts_to_plane_distance(self, model, points):
        distance = []
        A, B, C, D = model
        x, y, z = points.T
        denominator = math.sqrt(A ** 2 + B ** 2 + C ** 2)
        for i in range(x.shape[0]):
            numerator = abs(A * x[i] + B * y[i] + C * z[i] + D)
            distance.append(numerator / denominator)
        distance = np.nan_to_num(distance, nan=0.0, posinf=0.0, neginf=0.0)
        return distance
          

if __name__ == '__main__':
    Analysis().Calculate_data('/home/demo/Downloads/2023_07_04_14_40_1010ref_0_120m.bag', [0, 100], [0.65, 2.35, -1.87, -0.35, 119.9, 120.15], [], [0, 0, 0, 1, 0], 0)
    print('What have done is done')