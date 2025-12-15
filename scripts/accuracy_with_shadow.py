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


def load_shadow_mask(mask_path):
    """
    加载阴影掩膜图片

    参数:
    mask_path: 阴影掩膜图片路径

    返回:
    shadow_mask: 阴影区域掩膜（255表示阴影区域）
    non_shadow_mask: 非阴影区域掩膜（255表示非阴影区域）
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"无法读取掩膜图片: {mask_path}")

    # 创建阴影区域掩膜（原掩膜中非零部分为阴影区域）
    shadow_mask = (mask == 0).astype(np.uint8) * 255

    # 创建非阴影区域掩膜（原掩膜中零部分为非阴影区域）
    non_shadow_mask = (mask > 0).astype(np.uint8) * 255

    return shadow_mask, non_shadow_mask


def detect_color_region(img, target_color='#008000', color_tolerance=30):
    """
    检测图片中特定颜色区域

    参数:
    img: 输入图片
    target_color: 目标颜色（十六进制）
    color_tolerance: 颜色容差

    返回:
    color_mask: 颜色区域掩膜
    """
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 转换目标颜色为HSV
    target_bgr = hex_to_bgr(target_color)
    target_bgr_array = np.uint8([[target_bgr]])
    target_hsv = cv2.cvtColor(target_bgr_array, cv2.COLOR_BGR2HSV)[0][0]

    # 定义颜色范围
    lower_color = np.array([max(0, target_hsv[0] - color_tolerance), 50, 50])
    upper_color = np.array([min(179, target_hsv[0] + color_tolerance), 255, 255])

    # 创建颜色区域掩码
    color_mask = cv2.inRange(hsv, lower_color, upper_color)

    # 优化掩码
    kernel = np.ones((3, 3), np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

    return color_mask


def make_background_transparent(img, mask=None):
    """
    将图片黑色背景变为透明

    参数:
    img: 输入BGR图片
    mask: 可选，指定哪些区域应该不透明（白色为不透明区域）

    返回:
    transparent_img: BGRA格式的透明背景图片
    """
    # 将BGR转换为BGRA（添加Alpha通道）
    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    if mask is not None:
        # 使用提供的掩膜作为Alpha通道
        bgra[:, :, 3] = mask
    else:
        # 自动检测：将黑色背景设为透明
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, alpha_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        bgra[:, :, 3] = alpha_mask

    return bgra


def calculate_differences_by_region(img1, img2, shadow_mask, target_color='#377E22', color_tolerance=30):
    """
    分别计算阴影区域和非阴影区域的差异

    参数:
    img1: 第一张图片
    img2: 第二张图片
    shadow_mask: 阴影区域掩膜
    target_color: 目标颜色
    color_tolerance: 颜色容差

    返回:
    results: 包含各种差异信息的字典
    """
    # 确保掩膜与图片尺寸相同
    if shadow_mask.shape[:2] != img1.shape[:2]:
        shadow_mask = cv2.resize(shadow_mask, (img1.shape[1], img1.shape[0]))

    # 创建非阴影区域掩膜
    non_shadow_mask = cv2.bitwise_not(shadow_mask)

    # 检测两张图片中的目标颜色区域
    color_mask1 = detect_color_region(img1, target_color, color_tolerance)
    color_mask2 = detect_color_region(img2, target_color, color_tolerance)

    # 分别计算阴影区域和非阴影区域的差异
    shadow_diff = cv2.absdiff(
        cv2.bitwise_and(color_mask1, shadow_mask),
        cv2.bitwise_and(color_mask2, shadow_mask)
    )

    non_shadow_diff = cv2.absdiff(
        cv2.bitwise_and(color_mask1, non_shadow_mask),
        cv2.bitwise_and(color_mask2, non_shadow_mask)
    )

    # 计算整体差异
    total_diff = cv2.absdiff(color_mask1, color_mask2)

    # 计算阴影区域的差异统计
    shadow_pixels_total = np.count_nonzero(
        cv2.bitwise_or(
            cv2.bitwise_and(color_mask1, shadow_mask),
            cv2.bitwise_and(color_mask2, shadow_mask)
        )
    )
    shadow_pixels_diff = np.count_nonzero(shadow_diff)

    # 计算非阴影区域的差异统计
    non_shadow_pixels_total = np.count_nonzero(
        cv2.bitwise_or(
            cv2.bitwise_and(color_mask1, non_shadow_mask),
            cv2.bitwise_and(color_mask2, non_shadow_mask)
        )
    )
    non_shadow_pixels_diff = np.count_nonzero(non_shadow_diff)

    # 计算整体差异统计
    total_color_pixels = np.count_nonzero(color_mask1 | color_mask2)
    total_diff_pixels = np.count_nonzero(total_diff)

    # 计算百分比
    shadow_diff_percentage = (shadow_pixels_diff / shadow_pixels_total * 100) if shadow_pixels_total > 0 else 0
    non_shadow_diff_percentage = (
            non_shadow_pixels_diff / non_shadow_pixels_total * 100) if non_shadow_pixels_total > 0 else 0
    total_diff_percentage = (total_diff_pixels / total_color_pixels * 100) if total_color_pixels > 0 else 0

    # 创建阴影区域差异可视化（黄色表示差异）
    shadow_visual = np.zeros_like(img1)
    shadow_diff_colored = cv2.cvtColor(shadow_diff, cv2.COLOR_GRAY2BGR)
    shadow_diff_colored[shadow_diff == 255] = [0, 180, 180]  # BGR格式的黄色
    shadow_visual = cv2.addWeighted(shadow_visual, 1, shadow_diff_colored, 1.0, 0)

    # 将阴影区域差异图背景设为透明
    shadow_alpha_mask = cv2.bitwise_or(shadow_diff,
                                       cv2.cvtColor(shadow_visual, cv2.COLOR_BGR2GRAY))
    shadow_visual_transparent = make_background_transparent(shadow_visual, shadow_alpha_mask)

    # 创建非阴影区域差异可视化（蓝色表示差异）
    non_shadow_visual = np.zeros_like(img1)
    non_shadow_diff_colored = cv2.cvtColor(non_shadow_diff, cv2.COLOR_GRAY2BGR)
    non_shadow_diff_colored[non_shadow_diff == 255] = [255, 0, 0]  # BGR格式的蓝色
    non_shadow_visual = cv2.addWeighted(non_shadow_visual, 1, non_shadow_diff_colored, 1.0, 0)

    # 将非阴影区域差异图背景设为透明
    non_shadow_alpha_mask = cv2.bitwise_or(non_shadow_diff,
                                           cv2.cvtColor(non_shadow_visual, cv2.COLOR_BGR2GRAY))
    non_shadow_visual_transparent = make_background_transparent(non_shadow_visual, non_shadow_alpha_mask)

    # 创建整体差异可视化（红色表示阴影差异，蓝色表示非阴影差异）
    total_visual = np.zeros_like(img1)
    total_visual = cv2.addWeighted(total_visual, 1, shadow_diff_colored, 0.7, 0)
    total_visual = cv2.addWeighted(total_visual, 1, non_shadow_diff_colored, 0.7, 0)

    # 将整体差异图背景设为透明
    total_alpha_mask = cv2.bitwise_or(cv2.cvtColor(total_visual, cv2.COLOR_BGR2GRAY),
                                      cv2.bitwise_or(shadow_diff, non_shadow_diff))
    total_visual_transparent = make_background_transparent(total_visual, total_alpha_mask)

    return {
        'shadow_diff_percentage': shadow_diff_percentage,
        'non_shadow_diff_percentage': non_shadow_diff_percentage,
        'total_diff_percentage': total_diff_percentage,
        'shadow_diff_mask': shadow_diff,
        'non_shadow_diff_mask': non_shadow_diff,
        'total_diff_mask': total_diff,
        'shadow_visual': shadow_visual_transparent,  # 阴影区域差异图（透明背景）
        'non_shadow_visual': non_shadow_visual_transparent,  # 非阴影区域差异图（透明背景）
        'total_visual': total_visual_transparent,  # 整体差异图（透明背景）
        'shadow_pixels_total': shadow_pixels_total,
        'shadow_pixels_diff': shadow_pixels_diff,
        'non_shadow_pixels_total': non_shadow_pixels_total,
        'non_shadow_pixels_diff': non_shadow_pixels_diff,
        'color_mask1': color_mask1,
        'color_mask2': color_mask2
    }


def save_results(results, base_path):
    """
    保存差异分析结果

    参数:
    results: 差异计算结果
    base_path: 保存路径前缀
    """
    # 确保保存目录存在
    os.makedirs(os.path.dirname(base_path), exist_ok=True)

    # 1. 保存阴影区域差异图（透明背景）
    shadow_visual_path = f"{base_path}_shadow_difference.png"  # 使用PNG支持透明
    cv2.imwrite(shadow_visual_path, results['shadow_visual'])
    print(f"  ✓ 阴影区域差异图已保存: {shadow_visual_path} (透明背景)")

    # 2. 保存非阴影区域差异图（透明背景）
    non_shadow_visual_path = f"{base_path}_non_shadow_difference.png"  # 使用PNG支持透明
    cv2.imwrite(non_shadow_visual_path, results['non_shadow_visual'])
    print(f"  ✓ 非阴影区域差异图已保存: {non_shadow_visual_path} (透明背景)")

    # 3. 保存整体差异图（透明背景）
    total_visual_path = f"{base_path}_total_difference.png"  # 使用PNG支持透明
    cv2.imwrite(total_visual_path, results['total_visual'])
    print(f"  ✓ 整体差异图已保存: {total_visual_path} (透明背景)")


def main():

    img1_folder = r"D:\data\outputdir\new york\segmentation\new york\256_19_8\process"
    img2_folder = r"D:\data\outputdir\new york\segmentation\new york\256_19_8\ground_truth"
    shadow_mask_folder = r"D:\data\outputdir\new york\segmentation\new york\256_19_8\shadow"
    output_folder = r"D:\data\outputdir\new york\segmentation\new york\256_19_8\differences2"

    os.makedirs(output_folder, exist_ok=True)

    TARGET_COLOR = "#008000"
    COLOR_TOLERANCE = 20

    # 支持的图片格式
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

    print("=" * 60)
    print("批量区域差异分析工具（透明背景版）")
    print("=" * 60)
    print(f"图片1文件夹: {img1_folder}")
    print(f"图片2文件夹: {img2_folder}")
    print(f"阴影掩膜文件夹: {shadow_mask_folder}")
    print(f"输出文件夹: {output_folder}")
    print(f"目标颜色: {TARGET_COLOR}")
    print(f"颜色容差: {COLOR_TOLERANCE}")
    print("=" * 60)

    # 获取第一个文件夹中的所有图片
    img1_files = [f for f in os.listdir(img1_folder)
                  if os.path.splitext(f)[1].lower() in image_extensions]

    if not img1_files:
        print("错误: 第一个文件夹中没有找到支持的图片文件")
        return

    print(f"找到 {len(img1_files)} 个待处理图片")
    print("=" * 60)

    # 用于统计的列表
    all_shadow_diff = []
    all_non_shadow_diff = []
    all_total_diff = []

    # 用于跟踪处理的文件数
    processed_count = 0
    error_count = 0
    skipped_count = 0

    # 批量处理所有图片
    for filename in img1_files:
        try:
            print(f"\n[{processed_count + 1}/{len(img1_files)}] 处理文件: {filename}")

            # 构建所有文件路径（假设文件名完全相同）
            img1_path = os.path.join(img1_folder, filename)
            img2_path = os.path.join(img2_folder, filename)
            shadow_mask_path = os.path.join(shadow_mask_folder, filename)

            # 检查所有文件是否存在
            missing_files = []
            if not os.path.exists(img2_path):
                missing_files.append(f"图片2: {filename}")
            if not os.path.exists(shadow_mask_path):
                missing_files.append(f"阴影掩膜: {filename}")

            if missing_files:
                print(f"  跳过: 文件缺失 - {', '.join(missing_files)}")
                skipped_count += 1
                continue

            # 1. 加载图片
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            shadow_mask_img = cv2.imread(shadow_mask_path, cv2.IMREAD_GRAYSCALE)

            if img1 is None or img2 is None or shadow_mask_img is None:
                print(f"  错误: 无法读取图片文件")
                error_count += 1
                continue

            print(f"  图片1尺寸: {img1.shape}")
            print(f"  图片2尺寸: {img2.shape}")
            print(f"  阴影掩膜尺寸: {shadow_mask_img.shape}")

            # 2. 加载阴影掩膜
            shadow_mask, non_shadow_mask = load_shadow_mask(shadow_mask_path)

            # 3. 计算区域差异
            results = calculate_differences_by_region(
                img1, img2, shadow_mask,
                TARGET_COLOR, COLOR_TOLERANCE
            )

            # 4. 收集统计信息
            all_shadow_diff.append(results['shadow_diff_percentage'])
            all_non_shadow_diff.append(results['non_shadow_diff_percentage'])
            all_total_diff.append(results['total_diff_percentage'])

            # 5. 保存结果（使用原始文件名作为基础）
            base_name = os.path.splitext(filename)[0]
            output_base_path = os.path.join(output_folder, base_name)

            # 修改save_results函数以接受自定义路径
            save_batch_results(results, output_base_path)

            # 显示当前结果
            print(f"  阴影区域差异: {results['shadow_diff_percentage']:.2f}%")
            print(f"  非阴影区域差异: {results['non_shadow_diff_percentage']:.2f}%")
            print(f"  整体差异: {results['total_diff_percentage']:.2f}%")

            processed_count += 1

        except Exception as e:
            print(f"  处理失败: {str(e)}")
            error_count += 1
            continue


    print(f"\nsave to: {output_folder}")
    print("=" * 60)


def save_batch_results(results, base_path):
    # 确保保存目录存在
    os.makedirs(os.path.dirname(base_path), exist_ok=True)

    # 1. 保存阴影区域差异图（透明背景）
    shadow_visual_path = f"{base_path}_shadow_difference.png"  # 使用PNG支持透明
    cv2.imwrite(shadow_visual_path, results['shadow_visual'])

    # 2. 保存非阴影区域差异图（透明背景）
    non_shadow_visual_path = f"{base_path}_non_shadow_difference.png"  # 使用PNG支持透明
    cv2.imwrite(non_shadow_visual_path, results['non_shadow_visual'])

    # 3. 保存整体差异图（透明背景）
    total_visual_path = f"{base_path}_total_difference.png"  # 使用PNG支持透明
    cv2.imwrite(total_visual_path, results['total_visual'])


# 运行主函数
if __name__ == "__main__":
    main()