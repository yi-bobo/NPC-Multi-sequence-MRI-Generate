import os
import random
import numpy as np
from PIL import Image
import nibabel as nib    

class output_Nii_Png:
    def __init__(self, opt):
        self.sour_name = opt.data.sour_img_name
        self.targ_name = opt.data.targ_img_name

    def plot(self, input, target, pred_target, data_mapping, path):
            # 保存为nii
            for key, value in data_mapping.items():
                value = value.cpu().squeeze(0).squeeze(0).numpy()
                self.save_nii(value, os.path.join(path, f"{key}.nii"))
            self.save_nii(input, os.path.join(path, f"{self.sour_name}.nii"))
            self.save_nii(target, os.path.join(path, f"{self.targ_name}.nii"))
            self.save_nii(pred_target, os.path.join(path, f"pred_{self.targ_name}.nii"))

            # 保存为png
            nonzero_slices = []
            for slice in range(pred_target.shape[0]):
                if target[slice, :, :].sum() > 0:
                    nonzero_slices.append(slice)
            if len(nonzero_slices) != 0:
                slice = random.randint(0, len(nonzero_slices)-1)
                slice = nonzero_slices[slice]
                self.save_png(self.norm_png(input[slice, :, :]), os.path.join(path, f"{self.sour_name}_{slice}.png"))
                self.save_png(self.norm_png(target[slice, :, :]), os.path.join(path, f"{self.targ_name}_{slice}.png"))
                self.save_png(self.norm_png(pred_target[slice, :, :]), os.path.join(path, f"pred_{self.targ_name}_{slice}.png"))

    def norm_png(self, data):
        data = (data - data.min()) / (data.max() - data.min())
        return data

    def save_png(self, data, path):
        data = (data*255).astype(np.uint8)
        data = Image.fromarray(data)
        data.save(path)

    def save_nii(self, data, path):
        data = np.transpose(data, (2, 1, 0))
        data = np.rot90(data, k=-2, axes=(1, 2))  # 顺时针旋转 180 度
        voxel_spacing = [1, 1, 6]  # z, x, y
        affine = np.diag(voxel_spacing + [1])
        nii_x = nib.Nifti1Image(data, affine)
        nib.save(nii_x, path)
    
    def plot_mid_sample(self, sour_data, targ_data, pred_x_mid_sample, path):
        # 保存为png
        path = os.path.join(path, f"{self.sour_name}_to_{self.targ_name}.png")
        nonzero_slices = []
        for slice in range(sour_data.shape[0]):
            if targ_data[slice, :, :].sum() > 0:
                nonzero_slices.append(slice)
        if len(nonzero_slices) != 0:
            slice = random.randint(0, len(nonzero_slices)-1)
            slice = nonzero_slices[slice]
            img1 = self.norm_png(sour_data[slice, :, :])
            img2 = self.norm_png(targ_data[slice, :, :])
            img3 = self.norm_png(pred_x_mid_sample[0][slice, :, :])
            img4 = self.norm_png(pred_x_mid_sample[1][slice, :, :])
            img5 = self.norm_png(pred_x_mid_sample[2][slice, :, :])
            img6 = self.norm_png(pred_x_mid_sample[3][slice, :, :])
            self.save_combined_image([img1, img2, img3, img4, img5, img6], path, grid_size=(1, 6))

    def save_combined_image(self, img_list, output_path, grid_size=(1, 6)):
        # 检查输入
        assert len(img_list) == grid_size[0] * grid_size[1], "图片数量与网格大小不匹配！"
        # 获取单张图片的高度和宽度
        height, width = img_list[0].shape
        # 创建空白画布，用于拼接
        grid_height = height * grid_size[0]  # 总高度
        grid_width = width * grid_size[1]   # 总宽度
        combined_image = np.zeros((grid_height, grid_width), dtype=np.uint8)  # 假设图片是灰度图
        # 将每张图片放到画布上的对应位置
        for idx, img in enumerate(img_list):
            row, col = divmod(idx, grid_size[1])  # 计算图片的行、列位置
            y_start, y_end = row * height, (row + 1) * height
            x_start, x_end = col * width, (col + 1) * width
            combined_image[y_start:y_end, x_start:x_end] = img
        # 将拼接后的 NumPy 数组转换为图片并保存
        final_image = Image.fromarray(combined_image)
        final_image.save(output_path)