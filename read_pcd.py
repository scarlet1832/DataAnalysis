import numpy as np
import pandas as pd
import re
from io import StringIO

numpy_pcd_type_mappings = [(np.dtype('float32'), ('F', 4)),
                           (np.dtype('float64'), ('F', 8)),
                           (np.dtype('uint8'), ('U', 1)),
                           (np.dtype('uint16'), ('U', 2)),
                           (np.dtype('uint32'), ('U', 4)),
                           (np.dtype('uint64'), ('U', 8)),
                           (np.dtype('int16'), ('I', 2)),
                           (np.dtype('int32'), ('I', 4)),
                           (np.dtype('int64'), ('I', 8))]

pcd_type_to_numpy_type = dict((q, p) for (p, q) in numpy_pcd_type_mappings)


class pcd_cls(object):
    @staticmethod
    def parse_header(lines):
        metadata = {}
        for ln in lines:
            if ln.startswith('#') or len(ln) < 2:
                continue
            match = re.match('(\w+)\s+([\w\s\.]+)', ln)
            if not match:
                # warnings.warn("warning: can't understand line: %s" % ln)
                continue
            key, value = match.group(1).lower(), match.group(2)
            if key == 'version':
                metadata[key] = value
            elif key in ('fields', 'type'):
                metadata[key] = value.split()
            elif key in ('size', 'count'):
                metadata[key] = map(int, value.split())
            elif key in ('width', 'height', 'points'):
                metadata[key] = int(value)
            elif key == 'viewpoint':
                metadata[key] = map(float, value.split())
            elif key == 'data':
                metadata[key] = value.strip().lower()
            # TODO apparently count is not required?
        # add some reasonable defaults
        if 'count' not in metadata:
            metadata['count'] = [1] * len(metadata['fields'])
        if 'viewpoint' not in metadata:
            metadata['viewpoint'] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        if 'version' not in metadata:
            metadata['version'] = '.7'
        return metadata

    @staticmethod
    def read_pcd(filename: str,output_filename: str):
        pc_data = np.array([[]])
        data = pd.DataFrame()
        fields = []
        with open(filename, 'rb') as f:
            header = []
            while True:
                ln = f.readline().strip().decode()
                header.append(ln)
                if ln.startswith('DATA'):
                    metadata = pcd_cls.parse_header(header)
                    dtype = pcd_cls.build_dtype(metadata)
                    # print(dtype)
                    break

            if metadata['fields']:
                fields = metadata['fields']

            if metadata['data'] == 'binary':
                rowstep = metadata['points'] * dtype.itemsize
                buf = f.read(rowstep)
                pc_data = np.frombuffer(buf, dtype=dtype)
                data = pd.DataFrame(columns=fields, data=pc_data)               
            elif metadata['data'] == 'ascii':
                rowstep = metadata['points'] * dtype.itemsize
                buf = f.read(rowstep)
                buf = buf.decode('utf-8').strip().strip("\n")
                csv_data = StringIO(buf)
                data = np.array(pd.read_csv(csv_data,header=None,sep=' ').values.tolist())
                data = pd.DataFrame(columns=fields, data=data)
        data.to_excel(output_filename,index=False)
        print("Data written to Excel successfully!")
      
    @staticmethod
    def build_dtype(metadata):
        """ build numpy structured array dtype from pcl metadata.
        note that fields with count > 1 are 'flattened' by creating multiple
        single-count fields.
        TODO: allow 'proper' multi-count fields.
        """
        fieldnames = []
        typenames = []
        for f, c, t, s in zip(metadata['fields'],
                              metadata['count'],
                              metadata['type'],
                              metadata['size']):
            np_type = pcd_type_to_numpy_type[(t, s)]
            if c == 1:
                fieldnames.append(f)
                typenames.append(np_type)
            else:
                fieldnames.extend(['%s_%04d' % (f, i) for i in range(c)])
                typenames.extend([np_type] * c)

        dtype = np.dtype(list(zip(fieldnames, typenames)))
        return dtype
filename='/home/demo/Downloads/gt_corners.pcd'
output_filename='/home/demo/Downloads/gt_corners_ascii.xlsx'
pcd_cls.read_pcd(filename,output_filename)
# import time
# t = time.time()
# pcd = PCD2CSV.read_pcd("data.pcd")
# print(pcd)
# print(f'takes {time.time() -t}s')