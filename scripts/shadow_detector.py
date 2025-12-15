import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple


# 识别阴影区域
def calculate_mask(org_image: np.ndarray,
                   ab_threshold: int,
                   region_adjustment_kernel_size: int) -> np.ndarray:
    # 将原图转换为LAB空间图片
    lab_img = cv.cvtColor(org_image, cv.COLOR_BGR2LAB)

    # L以及A、B的取值范围
    l_range = (0, 100)
    ab_range = (-128, 127)

    # 调整参数
    lab_img = lab_img.astype('int16')
    lab_img[:, :, 0] = lab_img[:, :, 0] * l_range[1] / 255
    lab_img[:, :, 1] += ab_range[0]
    lab_img[:, :, 2] += ab_range[0]

    # 计算所有像素的LAB平均值及其阈值
    means = [np.mean(lab_img[:, :, i]) for i in range(3)]
    thresholds = [means[i] - (np.std(lab_img[:, :, i]) / 3) for i in range(3)]

    # mean(A) + mean(B)<threshold, 归类为阴影
    if sum(means[1:]) <= ab_threshold:
        mask = cv.inRange(lab_img, (l_range[0], ab_range[0], ab_range[0]),
                          (thresholds[0], ab_range[1], ab_range[1]))
    # 否则将L、B低于threshold的部分归类为阴影
    else:
        mask = cv.inRange(lab_img, (l_range[0], ab_range[0], ab_range[0]),
                          (thresholds[0], ab_range[1], thresholds[2]))

    kernel_size = (region_adjustment_kernel_size, region_adjustment_kernel_size)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, kernel_size)
    cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, mask)
    cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, mask)

    return mask


def identify_shadow_regions(org_image: np.ndarray,
                            ab_threshold: int = 256,
                            region_adjustment_kernel_size: int = 10,
                            verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    # 识别阴影区域
    mask = calculate_mask(org_image, ab_threshold, region_adjustment_kernel_size)

    # 将掩膜转换为RGB格式以便显示
    mask_rgb = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)

    # 如果verbose为True，显示原图和阴影区域
    if verbose:
        _, axes = plt.subplots(1, 2)
        ax = axes.ravel()

        plt.title("阴影区域识别结果")

        ax[0].imshow(cv.cvtColor(org_image, cv.COLOR_BGR2RGB))
        ax[0].set_title("原图")

        ax[1].imshow(cv.cvtColor(mask_rgb, cv.COLOR_BGR2RGB))
        ax[1].set_title("阴影区域掩膜")

        plt.tight_layout()
        plt.show()

    return mask, mask_rgb


# 主要处理函数
def process_image_file(img_name: str,
                       save: bool = False,
                       ab_threshold: int = 256,
                       region_adjustment_kernel_size: int = 10,
                       verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 读取图片
    org_image = cv.imread(img_name)
    if org_image is None:
        print(f"无法读取图片: {img_name}")
        return None, None, None

    print(f"读取图片: {img_name}")

    # 识别阴影区域
    mask, mask_rgb = identify_shadow_regions(org_image, ab_threshold,
                                             region_adjustment_kernel_size, verbose)

    # 如果需要保存结果
    if save:
        # 保存原图与掩膜叠加的效果
        overlay = cv.addWeighted(org_image, 0.7, mask_rgb, 0.3, 0)

        # 生成保存文件名
        if "." in img_name:
            f_name = img_name[:img_name.index(".")] + "_shadowMask" + img_name[img_name.index("."):]
        else:
            f_name = img_name + "_shadowMask.png"

        # 保存结果
        #cv.imwrite(f_name, mask_rgb)
        print(f"已保存阴影区域掩膜为: {f_name}")

        # 保存叠加效果
        overlay_name = f_name.replace("_shadowMask.", "_shadowOverlay.")
        #cv.imwrite(overlay_name, overlay)
        print(f"已保存叠加效果为: {overlay_name}")

    return org_image, mask, mask_rgb

import os

import os

# 使用示例
if __name__ == "__main__":
    # 设置输入和输出文件夹路径
    input_folder = r"D:\data\outputdir\new york\segmentation\new york\256_19_8\256_19_8"
    output_folder = r"D:\data\outputdir\new york\segmentation\new york\256_19_8\shadow"

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 批量处理所有图片
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(input_folder, filename)

            try:
                # 处理图片
                org_image, mask, mask_rgb = process_image_file(
                    input_path,
                    ab_threshold=4,
                    save=False,
                    verbose=False
                )

                if org_image is not None:
                    # 保存结果
                    base_name = os.path.splitext(filename)[0]
                    mask_path = os.path.join(output_folder, f"{base_name}_shadowMask.png")
                    overlay_path = os.path.join(output_folder, f"{base_name}_shadowOverlay.png")

                    # 黑白反转：阴影区域变白(255)，非阴影区域变黑(0)
                    # 方法1：直接反转掩膜
                    mask_inverted = cv.bitwise_not(mask)  # 反转二值图像
                    mask_rgb_inverted = cv.cvtColor(mask_inverted, cv.COLOR_GRAY2RGB)  # 转回RGB

                    # 保存反转后的掩膜
                    cv.imwrite(mask_path, mask_rgb_inverted)

                    # 用反转后的掩膜创建叠加效果
                    overlay = cv.addWeighted(org_image, 0.7, mask_rgb_inverted, 0.3, 0)
                    #cv.imwrite(overlay_path, overlay)

                    print(f"processed: {filename}")

            except Exception as e:
                print(f"failed {filename}: {e}")