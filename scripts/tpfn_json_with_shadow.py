import cv2
import numpy as np
import os
import json
from datetime import datetime


def hex_to_bgr(hex_color):
    """将十六进制颜色代码转换为BGR格式"""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)


def load_shadow_mask(shadow_path, gt_img_shape):
    """
    加载并处理阴影掩码图片

    参数:
    shadow_path: 阴影图片路径
    gt_img_shape: 基准图片形状（用于调整大小）

    返回:
    shadow_mask: 阴影区域掩码（黑色部分为阴影=1，白色部分为非阴影=0）
    non_shadow_mask: 非阴影区域掩码
    """
    shadow_img = cv2.imread(shadow_path, cv2.IMREAD_GRAYSCALE)

    # 调整阴影图片大小与基准图片一致
    if shadow_img.shape[:2] != gt_img_shape[:2]:
        shadow_img = cv2.resize(shadow_img, (gt_img_shape[1], gt_img_shape[0]))

    # 二值化：黑色部分（阴影）为1，白色部分（非阴影）为0
    _, shadow_binary = cv2.threshold(shadow_img, 127, 1, cv2.THRESH_BINARY_INV)

    shadow_mask = shadow_binary.astype(np.uint8)  # 阴影区域为1
    non_shadow_mask = 1 - shadow_mask  # 非阴影区域为1

    return shadow_mask, non_shadow_mask


def calculate_color_difference(img1, img2, target_color='#008000', color_tolerance=30):
    """
    计算两张图片中特定颜色区域的差异
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

    return colored_mask1, colored_mask2


def calculate_segmentation_metrics(gt_mask, pred_mask, region_mask=None):
    """
    计算语义分割的TP/FP/FN/TN指标，可指定区域掩码
    """
    # 确保掩码是二值化的
    gt_binary = (gt_mask == 255).astype(np.uint8)
    pred_binary = (pred_mask == 255).astype(np.uint8)

    # 如果指定了区域掩码，只计算该区域内的像素
    if region_mask is not None:
        # 确保区域掩码与GT掩码形状相同
        if region_mask.shape != gt_binary.shape:
            region_mask = cv2.resize(region_mask, (gt_binary.shape[1], gt_binary.shape[0]))

        # 创建区域内的有效掩码
        valid_region = (region_mask == 1)

        # 只在有效区域内计算
        gt_region = gt_binary[valid_region]
        pred_region = pred_binary[valid_region]

        # 计算区域内的混淆矩阵
        tp = np.sum((gt_region == 1) & (pred_region == 1))
        fp = np.sum((gt_region == 0) & (pred_region == 1))
        fn = np.sum((gt_region == 1) & (pred_region == 0))
        tn = np.sum((gt_region == 0) & (pred_region == 0))

        total_pixels = gt_region.size

    else:
        # 计算整个图像的混淆矩阵
        tp = np.sum((gt_binary == 1) & (pred_binary == 1))
        fp = np.sum((gt_binary == 0) & (pred_binary == 1))
        fn = np.sum((gt_binary == 1) & (pred_binary == 0))
        tn = np.sum((gt_binary == 0) & (pred_binary == 0))

        total_pixels = gt_binary.size

    # 计算指标
    metrics = {
        'TP': int(tp),
        'FP': int(fp),
        'FN': int(fn),
        'TN': int(tn),
        'Total_pixels': int(total_pixels),
        'Region_area': int(total_pixels)  # 区域面积
    }

    # 计算百分比
    if total_pixels > 0:
        metrics['TP_percentage'] = float(tp / total_pixels * 100)
        metrics['FP_percentage'] = float(fp / total_pixels * 100)
        metrics['FN_percentage'] = float(fn / total_pixels * 100)
        metrics['TN_percentage'] = float(tn / total_pixels * 100)

    # 计算准确率、召回率、F1分数
    if (tp + fp) > 0:
        metrics['Precision'] = float(tp / (tp + fp))
    else:
        metrics['Precision'] = 0.0

    if (tp + fn) > 0:
        metrics['Recall'] = float(tp / (tp + fn))
    else:
        metrics['Recall'] = 0.0

    precision = metrics['Precision']
    recall = metrics['Recall']
    if (precision + recall) > 0:
        metrics['F1_Score'] = float(2 * precision * recall / (precision + recall))
    else:
        metrics['F1_Score'] = 0.0

    return metrics


def calculate_overall_metrics(image_metrics_list, region_type='shadow'):
    """
    计算某个区域的总体指标（像素累加方式）

    参数:
    image_metrics_list: 所有图片的指标列表
    region_type: 区域类型 ('shadow' 或 'non_shadow')

    返回:
    overall_metrics: 总体指标字典
    """
    # 累加所有像素
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    total_area = 0

    # 收集各图片的指标用于计算平均
    precisions = []
    recalls = []
    f1_scores = []

    for img_metrics in image_metrics_list:
        region_data = img_metrics[f'{region_type}_region']

        # 累加像素
        total_tp += region_data['TP']
        total_fp += region_data['FP']
        total_fn += region_data['FN']
        total_tn += region_data['TN']
        total_area += region_data['area_pixels']

        # 收集指标
        precisions.append(region_data['Precision'])
        recalls.append(region_data['Recall'])
        f1_scores.append(region_data['F1_Score'])

    # 计算总体指标（像素累加）
    if (total_tp + total_fp) > 0:
        overall_precision = total_tp / (total_tp + total_fp)
    else:
        overall_precision = 0.0

    if (total_tp + total_fn) > 0:
        overall_recall = total_tp / (total_tp + total_fn)
    else:
        overall_recall = 0.0

    if (overall_precision + overall_recall) > 0:
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall)
    else:
        overall_f1 = 0.0

    # 计算平均指标（各图片平均）
    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0
    avg_f1 = np.mean(f1_scores) if f1_scores else 0

    # 计算标准差
    std_precision = np.std(precisions) if precisions else 0
    std_recall = np.std(recalls) if recalls else 0
    std_f1 = np.std(f1_scores) if f1_scores else 0

    return {
        'total_images': len(image_metrics_list),
        'total_area_pixels': int(total_area),
        'average_area_per_image': float(total_area / len(image_metrics_list)) if image_metrics_list else 0,

        # 像素累加指标
        'overall_TP': int(total_tp),
        'overall_FP': int(total_fp),
        'overall_FN': int(total_fn),
        'overall_TN': int(total_tn),
        'overall_Precision': float(overall_precision),
        'overall_Recall': float(overall_recall),
        'overall_F1_Score': float(overall_f1),

        # 平均指标
        'average_Precision': float(avg_precision),
        'average_Recall': float(avg_recall),
        'average_F1_Score': float(avg_f1),

        # 标准差
        'std_Precision': float(std_precision),
        'std_Recall': float(std_recall),
        'std_F1_Score': float(std_f1),

        # 百分比
        'overall_TP_percentage': float(total_tp / total_area * 100) if total_area > 0 else 0,
        'overall_FP_percentage': float(total_fp / total_area * 100) if total_area > 0 else 0,
        'overall_FN_percentage': float(total_fn / total_area * 100) if total_area > 0 else 0,
        'overall_TN_percentage': float(total_tn / total_area * 100) if total_area > 0 else 0
    }


def main():
    """主函数：批量对比两个文件夹中对应图片的差异，输出阴影和非阴影区域的指标"""

    # 设置文件夹路径
    folder1_path = r"D:\data\outputdir\new york\segmentation\new york\256_19_8\ground_truth"
    folder2_path = r"D:\data\outputdir\new york\segmentation\new york\256_19_8\process"
    shadow_folder_path = r"D:\data\outputdir\new york\segmentation\new york\256_19_8\shadow"
    output_folder = r"D:\data\outputdir\new york\segmentation\new york\256_19_8\metrics_output"

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # JSON输出文件路径
    json_output_path = os.path.join(output_folder, "shadow_region_metrics_with_summary.json")

    # 目标颜色
    TARGET_GREEN = "#008000"

    # 支持的图片格式
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

    # 获取第一个文件夹中的所有图片
    folder1_files = [f for f in os.listdir(folder1_path)
                     if os.path.splitext(f)[1].lower() in image_extensions]

    # 存储所有图片的指标
    all_metrics = {
        "metadata": {
            "project": "Road Segmentation Evaluation with Shadows",
            "ground_truth_folder": folder1_path,
            "predicted_folder": folder2_path,
            "shadow_folder": shadow_folder_path,
            "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "color_target": TARGET_GREEN
        },
        "image_metrics": [],
        "summary": {}
    }

    # 批量处理
    processed_count = 0
    image_metrics_list = []

    for filename in folder1_files:
        try:
            img1_path = os.path.join(folder1_path, filename)
            img2_path = os.path.join(folder2_path, filename)
            shadow_path = os.path.join(shadow_folder_path, filename)

            # 检查文件是否存在
            if not os.path.exists(img2_path) or not os.path.exists(shadow_path):
                continue

            # 读取图片
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)

            if img1 is None or img2 is None:
                continue

            # 加载阴影掩码
            shadow_mask, non_shadow_mask = load_shadow_mask(shadow_path, img1.shape)

            # 计算阴影和非阴影区域面积
            shadow_area = np.sum(shadow_mask)
            non_shadow_area = np.sum(non_shadow_mask)

            # 计算颜色掩码
            gt_mask, pred_mask = calculate_color_difference(
                img1, img2, TARGET_GREEN, color_tolerance=20
            )

            # 计算阴影区域的指标
            shadow_metrics = calculate_segmentation_metrics(gt_mask, pred_mask, shadow_mask)

            # 计算非阴影区域的指标
            non_shadow_metrics = calculate_segmentation_metrics(gt_mask, pred_mask, non_shadow_mask)

            # 存储图片的指标
            image_metric = {
                'filename': filename,
                'image_size': f"{img1.shape[1]}x{img1.shape[0]}",
                'shadow_region': {
                    'area_pixels': int(shadow_area),
                    'area_percentage': float(shadow_area / (shadow_area + non_shadow_area) * 100) if (
                                                                                                             shadow_area + non_shadow_area) > 0 else 0,
                    'TP': shadow_metrics['TP'],
                    'FP': shadow_metrics['FP'],
                    'FN': shadow_metrics['FN'],
                    'TN': shadow_metrics['TN'],
                    'TP_percentage': shadow_metrics.get('TP_percentage', 0),
                    'FP_percentage': shadow_metrics.get('FP_percentage', 0),
                    'FN_percentage': shadow_metrics.get('FN_percentage', 0),
                    'TN_percentage': shadow_metrics.get('TN_percentage', 0),
                    'Precision': shadow_metrics.get('Precision', 0),
                    'Recall': shadow_metrics.get('Recall', 0),
                    'F1_Score': shadow_metrics.get('F1_Score', 0)
                },
                'non_shadow_region': {
                    'area_pixels': int(non_shadow_area),
                    'area_percentage': float(non_shadow_area / (shadow_area + non_shadow_area) * 100) if (
                                                                                                                 shadow_area + non_shadow_area) > 0 else 0,
                    'TP': non_shadow_metrics['TP'],
                    'FP': non_shadow_metrics['FP'],
                    'FN': non_shadow_metrics['FN'],
                    'TN': non_shadow_metrics['TN'],
                    'TP_percentage': non_shadow_metrics.get('TP_percentage', 0),
                    'FP_percentage': non_shadow_metrics.get('FP_percentage', 0),
                    'FN_percentage': non_shadow_metrics.get('FN_percentage', 0),
                    'TN_percentage': non_shadow_metrics.get('TN_percentage', 0),
                    'Precision': non_shadow_metrics.get('Precision', 0),
                    'Recall': non_shadow_metrics.get('Recall', 0),
                    'F1_Score': non_shadow_metrics.get('F1_Score', 0)
                }
            }

            all_metrics["image_metrics"].append(image_metric)
            image_metrics_list.append(image_metric)
            processed_count += 1

            # 打印当前图片的结果
            print(f"已处理: {filename}")
            print(f"  阴影区域 - TP: {shadow_metrics['TP']}, FP: {shadow_metrics['FP']}, FN: {shadow_metrics['FN']}, "
                  f"Precision: {shadow_metrics.get('Precision', 0):.4f}, Recall: {shadow_metrics.get('Recall', 0):.4f}")
            print(
                f"  非阴影区域 - TP: {non_shadow_metrics['TP']}, FP: {non_shadow_metrics['FP']}, FN: {non_shadow_metrics['FN']}, "
                f"Precision: {non_shadow_metrics.get('Precision', 0):.4f}, Recall: {non_shadow_metrics.get('Recall', 0):.4f}")

        except Exception as e:
            print(f"处理失败 {filename}: {str(e)}")

    # 计算总体统计summary
    if image_metrics_list:
        # 计算阴影区域的总体指标
        shadow_summary = calculate_overall_metrics(image_metrics_list, 'shadow')

        # 计算非阴影区域的总体指标
        non_shadow_summary = calculate_overall_metrics(image_metrics_list, 'non_shadow')

        # 计算全图总体指标（阴影+非阴影）
        total_area = shadow_summary['total_area_pixels'] + non_shadow_summary['total_area_pixels']
        total_tp = shadow_summary['overall_TP'] + non_shadow_summary['overall_TP']
        total_fp = shadow_summary['overall_FP'] + non_shadow_summary['overall_FP']
        total_fn = shadow_summary['overall_FN'] + non_shadow_summary['overall_FN']

        if (total_tp + total_fp) > 0:
            overall_precision = total_tp / (total_tp + total_fp)
        else:
            overall_precision = 0.0

        if (total_tp + total_fn) > 0:
            overall_recall = total_tp / (total_tp + total_fn)
        else:
            overall_recall = 0.0

        if (overall_precision + overall_recall) > 0:
            overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        else:
            overall_f1 = 0.0

        # 将summary添加到all_metrics
        all_metrics["summary"] = {
            "processed_images": processed_count,
            "total_images": len(image_metrics_list),

            # 阴影区域总结
            "shadow_region": shadow_summary,

            # 非阴影区域总结
            "non_shadow_region": non_shadow_summary,

            # 全图总体指标
            "overall_all_regions": {
                "total_TP": int(total_tp),
                "total_FP": int(total_fp),
                "total_FN": int(total_fn),
                "overall_Precision": float(overall_precision),
                "overall_Recall": float(overall_recall),
                "overall_F1_Score": float(overall_f1)
            },

            # 区域比例
            "region_proportion": {
                "shadow_proportion": float(
                    shadow_summary['total_area_pixels'] / total_area * 100) if total_area > 0 else 0,
                "non_shadow_proportion": float(
                    non_shadow_summary['total_area_pixels'] / total_area * 100) if total_area > 0 else 0
            }
        }

        # 打印总结信息
        print(f"\n{'=' * 60}")
        print("SUMMARY STATISTICS:")
        print(f"{'=' * 60}")
        print(f"Processed Images: {processed_count}")
        print(f"\nShadow Region Summary:")
        print(f"  Total Area: {shadow_summary['total_area_pixels']} pixels")
        print(f"  Overall Precision: {shadow_summary['overall_Precision']:.4f}")
        print(f"  Overall Recall: {shadow_summary['overall_Recall']:.4f}")
        print(f"  Overall F1 Score: {shadow_summary['overall_F1_Score']:.4f}")
        print(f"\nNon-Shadow Region Summary:")
        print(f"  Total Area: {non_shadow_summary['total_area_pixels']} pixels")
        print(f"  Overall Precision: {non_shadow_summary['overall_Precision']:.4f}")
        print(f"  Overall Recall: {non_shadow_summary['overall_Recall']:.4f}")
        print(f"  Overall F1 Score: {non_shadow_summary['overall_F1_Score']:.4f}")
        print(f"\nOverall All Regions:")
        print(f"  Overall Precision: {overall_precision:.4f}")
        print(f"  Overall Recall: {overall_recall:.4f}")
        print(f"  Overall F1 Score: {overall_f1:.4f}")
        print(f"{'=' * 60}")

    # 保存JSON文件
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    print(f"\n处理完成！共处理 {processed_count} 对图片")
    print(f"指标已保存到: {json_output_path}")


# 运行主函数
if __name__ == "__main__":
    main()