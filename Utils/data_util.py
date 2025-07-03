import numpy as np
from scipy.ndimage.interpolation import zoom

# region 归一化函数
class Normalizer:
    def __init__(self, max_val, min_val):
        """
        初始化归一化和反归一化的最大值和最小值。
        :param max_val: 数据集的最大值
        :param min_val: 数据集的最小值
        """
        self.max = max_val
        self.min = min_val

    def _clip_data(self, data):
        """
        将数据裁剪到最大值和最小值范围内。
        :param data: 输入数据
        :return: 裁剪后的数据
        """
        return data.clip(min=self.min, max=self.max)

    def norm_global(self, data):
        """
        根据设置的最大最小值对数据进行归一化。
        超出范围的数据会在归一化前被裁剪。
        :param data: 需要归一化的数据
        :return: 归一化后的数据
        """
        # 对数据进行裁剪，确保在 [min, max] 范围内
        clipped_data = self._clip_data(data)
        # 归一化到 [0, 1] 区间
        norm_data = (clipped_data - self.min) / (self.max - self.min)
        # 如果需要确保归一化结果严格在 [0, 1] 范围内
        norm_data = norm_data.clip(min=0, max=1)
        return norm_data

    def denorm_global(self, norm_data):
        """
        根据设置的最大最小值对归一化后的数据进行反归一化。
        超出范围的归一化数据会被裁剪到 [0, 1]。
        :param norm_data: 需要反归一化的数据
        :return: 反归一化后的数据
        """
        # 确保归一化数据在 [0, 1] 区间内
        clipped_norm_data = norm_data.clip(min=0, max=1)
        data = clipped_norm_data * (self.max - self.min) + self.min
        return data
    
def clip_data(data, min_val, max_val):
    """
    将数据裁剪到最大值和最小值范围内。
    :param data: 输入数据
    :return: 裁剪后的数据
    """
    return data.clip(min=min_val, max=max_val)

def norm_global(data, max_val, min_val):
    """
    根据设置的最大最小值对数据进行归一化。
    超出范围的数据会在归一化前被裁剪。
    :param data: 需要归一化的数据
    :return: 归一化后的数据
    """
    # 对数据进行裁剪，确保在 [min, max] 范围内
    clipped_data = clip_data(data, min_val, max_val)
    # 归一化到 [0, 1] 区间
    norm_data = (clipped_data - min_val) / (max_val - min_val)
    # 如果需要确保归一化结果严格在 [0, 1] 范围内
    norm_data = norm_data.clip(min=0, max=1)
    return norm_data

def denorm_global(norm_data, max_val, min_val):     
    """
    根据设置的最大最小值对归一化后的数据进行反归一化。
    超出范围的归一化数据会被裁剪到 [0, 1]。
    :param norm_data: 需要反归一化的数据
    :return: 反归一化后的数据
    """
    # 确保归一化数据在 [0, 1] 区间内
    clipped_norm_data = norm_data.clip(min=0, max=1)
    data = clipped_norm_data * (max_val - min_val) + min_val
    return data

def norm_ln_1(data, max_value=None, min_value=None):
    """
    对数据归一化，使得数据在[0,1]之间
    """
    if max_value is None:
        max_value = np.max(data)
    if min_value is None:
        min_value = np.min(data)
    if max_value == min_value:
        return data
    else:
        norm_data = (data - min_value) / (max_value - min_value)
        return norm_data
    
def norm_ln_2(data, max_value=None, min_value=None):
    """
    对数据归一化，使得数据在[-1,1]之间
    """
    if max_value is None:
        max_value = np.max(data)
    if min_value is None:
        min_value = np.min(data)
    if max_value == min_value:
        return data
    else:
        norm_data = (data - min_value) / (max_value - min_value) * 2 - 1
        return norm_data
    
def norm_log_1(data, max_value=None, min_value=None):
    """
    对数据归一化，使得数据的自然对数在[0,1]之间
    """
    if max_value is None:
        max_value = np.max(data)
    if min_value is None:
        min_value = np.min(data)
    norm_data = np.log(data - min_value + 1)
    norm_data = norm_data / max_value
    return norm_data

def norm_log_2(data, max_value=None, min_value=None):
    """
    对数据归一化，使得数据的自然对数在[-1,1]之间
    """
    if max_value is None:
        max_value = np.max(data)
    if min_value is None:
        min_value = np.min(data)
    norm_data = np.log(data - min_value + 1)
    norm_data = norm_data / max_value * 2 - 1
    return norm_data

def norm_mean_1(data, max_value=None, min_value=None):
    """
    均值归一化，将数据集中在 0 附近，并且通常落在 [-1, 1] 之间
    """
    if max_value is None:
        max_value = np.max(data)
    if min_value is None:
        min_value = np.min(data)
    mean = np.mean(data)
    if max_value == min_value:
        return data
    else:
        norm_data = (data - mean) / (max_value - min_value)
        return norm_data
    
# endregion

# region nii数据·读取·resize
def resize_arr(data, resize_shape):
    """
    将数据resize到指定的大小
    """
    data_shape = data.shape
    resize_data = zoom(data, (resize_shape[0] / data_shape[0], resize_shape[1] / data_shape[1], resize_shape[2] / data_shape[2]), order=1)
    return resize_data

def read_nii_data(nii_path):
    """
    读取nii数据，返回数据和头信息
    """
    import nibabel as nib
    nii_data = nib.load(nii_path)
    nii_header = nii_data.header
    nii_affine = nii_header.get_best_affine()
    nii_data = nii_data.get_fdata()
    return nii_data, nii_affine

# endregion