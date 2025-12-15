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


def make_background_transparent(img, diff_mask):
    """
    将图片黑色背景变为透明，红色差异区域保持

    参数:
    img: 原始BGR图片
    diff_mask: 差异掩码（白色为差异区域）

    返回:
    transparent_img: RGBA格式的透明背景图片
    """
    # 将BGR转换为BGRA（添加Alpha通道）
    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    # 创建差异区域的掩码（红色区域）
    diff_red_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    diff_red_mask[diff_mask == 255] = 255

    # 创建非黑色区域的掩码
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, non_black_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # 合并掩码：差异区域 + 非黑色区域
    combined_mask = cv2.bitwise_or(diff_red_mask, non_black_mask)

    # 将非差异区域的黑色背景设为透明
    bgra[:, :, 3] = combined_mask

    return bgra


def calculate_color_difference(img1, img2, target_color='#377E22', color_tolerance=30):
    """
    计算两张图片中特定颜色区域的差异

    参数:
    img1: 第一张图片
    img2: 第二张图片
    target_color: 目标颜色
    color_tolerance: 颜色容差

    返回:
    diff_percentage: 差异百分比
    diff_mask: 差异掩码
    colored_mask1: 第一张图颜色区域掩码
    colored_mask2: 第二张图颜色区域掩码
    """

    # 确保两张图片尺寸相同
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # 转换为HSV颜色空间
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    # 转换目标颜色为HSV
    target_bgr = hex_to_bgr(target_color)
    target_bgr_array = np.uint8([[target_bgr]])
    target_hsv = cv2.cvtColor(target_bgr_array, cv2.COLOR_BGR2HSV)[0][0]

    # 定义颜色范围
    lower_color = np.array([max(0, target_hsv[0] - color_tolerance), 50, 50])
    upper_color = np.array([min(179, target_hsv[0] + color_tolerance), 255, 255])

    # 创建颜色区域掩码
    colored_mask1 = cv2.inRange(hsv1, lower_color, upper_color)
    colored_mask2 = cv2.inRange(hsv2, lower_color, upper_color)

    # 优化掩码
    kernel = np.ones((3, 3), np.uint8)
    colored_mask1 = cv2.morphologyEx(colored_mask1, cv2.MORPH_CLOSE, kernel)
    colored_mask2 = cv2.morphologyEx(colored_mask2, cv2.MORPH_CLOSE, kernel)

    # 计算颜色区域的差异
    color_diff = cv2.absdiff(colored_mask1, colored_mask2)

    # 计算差异百分比
    total_color_pixels = np.count_nonzero(colored_mask1 | colored_mask2)
    different_pixels = np.count_nonzero(color_diff)

    if total_color_pixels == 0:
        diff_percentage = 0
    else:
        diff_percentage = (different_pixels / total_color_pixels) * 100

    return diff_percentage, color_diff, colored_mask1, colored_mask2


def main():
    """主函数：批量对比两个文件夹中对应图片的差异（简化版）"""

    # 设置文件夹路径
    folder1_path = r"D:\data\outputdir\new york\segmentation\new york\256_19_8\ground_truth"
    folder2_path = r"D:\data\outputdir\new york\segmentation\new york\256_19_8\process"
    output_folder = r"D:\data\outputdir\new york\segmentation\new york\256_19_8\differences"

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 目标颜色
    TARGET_GREEN = "#008000"

    # 支持的图片格式
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

    # 获取第一个文件夹中的所有图片
    folder1_files = [f for f in os.listdir(folder1_path)
                     if os.path.splitext(f)[1].lower() in image_extensions]

    # 批量处理
    results = []
    for filename in folder1_files:
        try:
            img1_path = os.path.join(folder1_path, filename)
            img2_path = os.path.join(folder2_path, filename)

            # 检查第二个文件夹中是否有对应文件
            if not os.path.exists(img2_path):
                print(f"跳过 {filename}: 第二个文件夹中无对应文件")
                continue

            print(f"处理: {filename}")

            # 读取图片
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)

            if img1 is None or img2 is None:
                print(f"  读取图片失败")
                continue

            # 计算差异
            diff_percentage, diff_mask, _, _ = calculate_color_difference(
                img1, img2, TARGET_GREEN, color_tolerance=20
            )

            results.append(diff_percentage)

            # 将差异区域标记为红色
            diff_visual = cv2.cvtColor(diff_mask, cv2.COLOR_GRAY2BGR)
            diff_visual[diff_mask == 255] = [0, 0, 255]  # BGR格式的红色

            # 将黑色背景变为透明
            transparent_img = make_background_transparent(diff_visual, diff_mask)

            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_folder, f"diff_{base_name}.png")  # 使用PNG格式支持透明
            cv2.imwrite(output_path, transparent_img)

            print(f"  差异: {diff_percentage:.2f}% (相似度: {100 - diff_percentage:.2f}%)")

        except Exception as e:
            print(f"  处理失败: {str(e)}")

    # 打印统计
    if results:
        print(f"\n处理完成！共处理 {len(results)} 对图片")
        print(f"平均差异: {np.mean(results):.2f}%")


# 运行主函数
if __name__ == "__main__":
    main()