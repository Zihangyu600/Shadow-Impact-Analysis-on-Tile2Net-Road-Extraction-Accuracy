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


def calculate_color_mask(img, target_color='#008000', color_tolerance=30):
    """
    计算图片中特定颜色区域的掩码

    参数:
    img: 输入图片
    target_color: 目标颜色
    color_tolerance: 颜色容差

    返回:
    mask: 颜色区域掩码
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
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # 优化掩码
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def calculate_segmentation_metrics(gt_mask, pred_mask):
    """
    计算语义分割的TP/FP/FN/TN指标

    参数:
    gt_mask: ground truth掩码
    pred_mask: predicted掩码

    返回:
    metrics: 包含各项指标的字典
    """
    # 确保掩码是二值化的
    gt_binary = (gt_mask == 255).astype(np.uint8)
    pred_binary = (pred_mask == 255).astype(np.uint8)

    # 计算混淆矩阵
    tp = np.sum((gt_binary == 1) & (pred_binary == 1))  # 真阳性
    fp = np.sum((gt_binary == 0) & (pred_binary == 1))  # 假阳性
    fn = np.sum((gt_binary == 1) & (pred_binary == 0))  # 假阴性
    tn = np.sum((gt_binary == 0) & (pred_binary == 0))  # 真阴性

    # 计算总面积
    total_pixels = gt_binary.size

    # 计算像素数
    gt_road_pixels = np.sum(gt_binary)
    pred_road_pixels = np.sum(pred_binary)

    # 计算准确率、召回率、F1分数
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # 计算百分比
    tp_percentage = (tp / total_pixels * 100) if total_pixels > 0 else 0
    fp_percentage = (fp / total_pixels * 100) if total_pixels > 0 else 0
    fn_percentage = (fn / total_pixels * 100) if total_pixels > 0 else 0
    tn_percentage = (tn / total_pixels * 100) if total_pixels > 0 else 0

    # 构建指标字典
    metrics = {
        'TP': int(tp),
        'FP': int(fp),
        'FN': int(fn),
        'TN': int(tn),
        'TP_percentage': float(tp_percentage),
        'FP_percentage': float(fp_percentage),
        'FN_percentage': float(fn_percentage),
        'TN_percentage': float(tn_percentage),
        'Total_pixels': int(total_pixels),
        'GT_road_pixels': int(gt_road_pixels),
        'Pred_road_pixels': int(pred_road_pixels),
        'Precision': float(precision),
        'Recall': float(recall),
        'F1_Score': float(f1_score)
    }

    return metrics


def main():
    """主函数：批量对比两个文件夹中对应图片的差异，只计算指标并保存为JSON"""

    # 设置文件夹路径
    folder1_path = r"D:\data\outputdir\new york\segmentation\new york\256_19_8\ground_truth"
    folder2_path = r"D:\data\outputdir\new york\segmentation\new york\256_19_8\process"
    output_folder = r"D:\data\outputdir\new york\segmentation\new york\256_19_8\metrics_output"

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # JSON输出文件路径
    json_output_path = os.path.join(output_folder, "segmentation_metrics.json")

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
            "project": "Road Segmentation Evaluation",
            "ground_truth_folder": folder1_path,
            "predicted_folder": folder2_path,
            "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "color_target": TARGET_GREEN,
            "color_tolerance": 20
        },
        "images": {},
        "summary": {}
    }

    # 存储每张图片的指标
    image_metrics = {}

    # 批量处理
    processed_count = 0
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

            # 确保两张图片尺寸相同
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

            # 计算颜色掩码
            gt_mask = calculate_color_mask(img1, TARGET_GREEN, color_tolerance=20)
            pred_mask = calculate_color_mask(img2, TARGET_GREEN, color_tolerance=20)

            # 计算语义分割指标
            metrics = calculate_segmentation_metrics(gt_mask, pred_mask)

            # 添加图片信息
            metrics['image_size'] = f"{img1.shape[1]}x{img1.shape[0]}"

            # 存储到字典中
            image_metrics[filename] = metrics
            processed_count += 1

            # 打印当前图片指标
            print(f"  TP: {metrics['TP']} ({metrics['TP_percentage']:.2f}%)")
            print(f"  FP: {metrics['FP']} ({metrics['FP_percentage']:.2f}%)")
            print(f"  FN: {metrics['FN']} ({metrics['FN_percentage']:.2f}%)")
            print(f"  准确率: {metrics['Precision']:.4f}")
            print(f"  召回率: {metrics['Recall']:.4f}")
            print(f"  F1分数: {metrics['F1_Score']:.4f}")

        except Exception as e:
            print(f"  处理失败: {str(e)}")

    # 存储所有图片的指标
    all_metrics["images"] = image_metrics

    # 计算总体统计（所有图片的均值）
    if image_metrics:
        precisions = [metrics['Precision'] for metrics in image_metrics.values()]
        recalls = [metrics['Recall'] for metrics in image_metrics.values()]
        f1_scores = [metrics['F1_Score'] for metrics in image_metrics.values()]

        # 计算总体像素统计（可选）
        total_tp = sum(metrics['TP'] for metrics in image_metrics.values())
        total_fp = sum(metrics['FP'] for metrics in image_metrics.values())
        total_fn = sum(metrics['FN'] for metrics in image_metrics.values())
        total_tn = sum(metrics['TN'] for metrics in image_metrics.values())
        total_pixels = sum(metrics['Total_pixels'] for metrics in image_metrics.values())

        # 计算总体指标
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (
                                                                                                                  overall_precision + overall_recall) > 0 else 0

        all_metrics["summary"] = {
            "processed_images": processed_count,
            "average_Precision": float(np.mean(precisions)),
            "average_Recall": float(np.mean(recalls)),
            "average_F1_Score": float(np.mean(f1_scores)),
            "std_Precision": float(np.std(precisions)),
            "std_Recall": float(np.std(recalls)),
            "std_F1_Score": float(np.std(f1_scores)),
            "overall_Precision": float(overall_precision),
            "overall_Recall": float(overall_recall),
            "overall_F1_Score": float(overall_f1),
            "total_TP": int(total_tp),
            "total_FP": int(total_fp),
            "total_FN": int(total_fn),
            "total_TN": int(total_tn)
        }

        # 保存JSON文件
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)

        print(f"\n处理完成！共处理 {processed_count} 对图片")
        print(f"JSON文件已保存到: {json_output_path}")
        print(f"\n统计摘要:")
        print(f"  平均准确率: {np.mean(precisions):.4f}")
        print(f"  平均召回率: {np.mean(recalls):.4f}")
        print(f"  平均F1分数: {np.mean(f1_scores):.4f}")
        print(f"  总体准确率: {overall_precision:.4f}")
        print(f"  总体召回率: {overall_recall:.4f}")
        print(f"  总体F1分数: {overall_f1:.4f}")
    else:
        print("没有找到可处理的图片")


# 运行主函数
if __name__ == "__main__":
    main()