import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def hex_to_bgr(hex_color):
    """将十六进制颜色代码转换为BGR格式"""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)


def detect_and_replace_red_to_green(img_path, target_green='#377E22'):
    """
    检测图片中的红色区域，并将其替换为目标绿色

    参数:
    img_path: 输入图片路径
    target_green: 目标绿色(十六进制)

    返回:
    modified_img: 修改后的图片（红色变为绿色）
    red_mask: 红色区域的掩码
    """
    # 读取图片
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图片: {img_path}")

    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 定义红色范围（HSV空间中红色有两个区域）
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # 创建红色掩码
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # 形态学操作优化掩码
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    # 转换为目标绿色
    target_bgr = hex_to_bgr(target_green)

    # 将红色区域替换为目标绿色
    modified_img = img.copy()
    modified_img[red_mask == 255] = target_bgr

    return modified_img, red_mask


def main():
    """主函数：批量修改文件夹中图片的人行道颜色"""

    # 设置文件夹路径
    input_folder = r"D:\data\outputdir\new york\segmentation\new york\256_19_8\seg_results"  # 输入文件夹
    output_folder = r"data/output"  # 输出文件夹

    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建输出文件夹: {output_folder}")

    # 目标颜色
    TARGET_GREEN = "#377E22"

    # 支持的图片格式
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    # 获取所有图片文件
    image_files = []
    for file in os.listdir(input_folder):
        if file.lower().endswith(supported_formats):
            image_files.append(file)

    print(f"找到 {len(image_files)} 张图片需要处理")
    print(f"输入文件夹: {input_folder}")
    print(f"输出文件夹: {output_folder}")

    # 批量处理图片
    processed_count = 0
    for i, img_file in enumerate(image_files, 1):
        # 构建完整路径
        img_path = os.path.join(input_folder, img_file)

        print(f"\n[{i}/{len(image_files)}] 处理: {img_file}")

        # 修改图片中的红色人行道为绿色
        modified_img, red_mask = detect_and_replace_red_to_green(img_path, TARGET_GREEN)

        # 生成输出文件名（保持原文件名）
        filename, ext = os.path.splitext(img_file)
        output_filename = f"{filename}"
        output_path = os.path.join(output_folder, output_filename)

        # 保存修改后的图片
        cv2.imwrite(output_path, modified_img)

        processed_count += 1

    # 总结
    print(f"save to: {output_folder}")

# 运行主函数
if __name__ == "__main__":
    main()