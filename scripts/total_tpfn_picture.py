import cv2
import numpy as np
import matplotlib.pyplot as plt
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


def make_background_transparent(img, mask):
    """
    将图片黑色背景变为透明

    参数:
    img: 原始BGR图片
    mask: 要保留的区域掩码

    返回:
    transparent_img: RGBA格式的透明背景图片
    """
    # 将BGR转换为BGRA（添加Alpha通道）
    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    # 将掩码区域设为不透明，其他区域设为透明
    bgra[:, :, 3] = mask

    return bgra


def calculate_color_difference(img1, img2, target_color='#008000', color_tolerance=30):
    """
    计算两张图片中特定颜色区域的差异

    参数:
    img1: 第一张图片（ground truth）
    img2: 第二张图片（predicted）
    target_color: 目标颜色
    color_tolerance: 颜色容差

    返回:
    colored_mask1: ground truth绿色区域掩码
    colored_mask2: predicted绿色区域掩码
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


def calculate_segmentation_metrics(gt_mask, pred_mask):
    """
    计算语义分割的TP/FP/FN/TN指标

    参数:
    gt_mask: ground truth掩码（图1）
    pred_mask: predicted掩码（图2）

    返回:
    metrics: 包含各项指标的字典
    """
    # 确保掩码是二值化的
    gt_binary = (gt_mask == 255).astype(np.uint8)
    pred_binary = (pred_mask == 255).astype(np.uint8)

    # 计算混淆矩阵
    tp = np.sum((gt_binary == 1) & (pred_binary == 1))  # 真阳性：两者都是道路
    fp = np.sum((gt_binary == 0) & (pred_binary == 1))  # 假阳性：预测有但真实没有
    fn = np.sum((gt_binary == 1) & (pred_binary == 0))  # 假阴性：真实有但预测没有
    tn = np.sum((gt_binary == 0) & (pred_binary == 0))  # 真阴性：两者都不是道路

    # 计算总面积
    total_pixels = gt_binary.size

    # 计算各类指标
    metrics = {
        'TP': int(tp),
        'FP': int(fp),
        'FN': int(fn),
        'TN': int(tn),
        'TP_pixels': int(tp),
        'FP_pixels': int(fp),
        'FN_pixels': int(fn),
        'TN_pixels': int(tn),
        'Total_pixels': int(total_pixels),
        'GT_road_pixels': int(np.sum(gt_binary)),
        'Pred_road_pixels': int(np.sum(pred_binary))
    }

    # 计算百分比
    if total_pixels > 0:
        metrics['TP_percentage'] = float(tp / total_pixels * 100)
        metrics['FP_percentage'] = float(fp / total_pixels * 100)
        metrics['FN_percentage'] = float(fn / total_pixels * 100)
        metrics['TN_percentage'] = float(tn / total_pixels * 100)

    # 计算准确率、召回率、F1分数
    if (tp + fp + fn) > 0:
        metrics['Precision'] = float(tp / (tp + fp) if (tp + fp) > 0 else 0)
        metrics['Recall'] = float(tp / (tp + fn) if (tp + fn) > 0 else 0)

        precision = metrics['Precision']
        recall = metrics['Recall']
        if (precision + recall) > 0:
            metrics['F1_Score'] = float(2 * precision * recall / (precision + recall))
        else:
            metrics['F1_Score'] = 0.0

    return metrics


def create_segmentation_visualization(gt_mask, pred_mask, metrics, visualization_type):
    """
    创建语义分割的可视化图像

    参数:
    gt_mask: ground truth掩码
    pred_mask: predicted掩码
    metrics: 指标字典
    visualization_type: 可视化类型 ('FN' 或 'FP')

    返回:
    visualization: 可视化图像
    """
    # 创建空白图像
    height, width = gt_mask.shape
    visualization = np.zeros((height, width, 3), dtype=np.uint8)

    if visualization_type == 'FN':
        # FN可视化：真实道路但预测不是（黄色）
        fn_mask = ((gt_mask == 255) & (pred_mask == 0)).astype(np.uint8) * 255
        if np.any(fn_mask == 255):
            visualization[fn_mask == 255] = [0, 255, 255]  # BGR黄色

    elif visualization_type == 'FP':
        # FP可视化：预测道路但真实不是（红色）
        fp_mask = ((gt_mask == 0) & (pred_mask == 255)).astype(np.uint8) * 255
        if np.any(fp_mask == 255):
            visualization[fp_mask == 255] = [0, 0, 255]  # BGR红色

    # 添加指标文本
    text_img = visualization.copy()
    text_y = 30

    # 添加标题
    cv2.putText(text_img, f"{visualization_type} Visualization", (10, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    text_y += 40

    # 添加指标信息
    cv2.putText(text_img, f"{visualization_type}: {metrics.get(f'{visualization_type}_pixels', 0)} pixels",
                (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    text_y += 25

    if f"{visualization_type}_percentage" in metrics:
        cv2.putText(text_img, f"{visualization_type}%: {metrics[f'{visualization_type}_percentage']:.2f}%",
                    (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return text_img


def main():
    """主函数：批量对比两个文件夹中对应图片的差异，输出TP/FP/FN指标"""

    # 设置文件夹路径
    folder1_path = r"D:\data\outputdir\new york\segmentation\new york\256_19_8\ground_truth"
    folder2_path = r"D:\data\outputdir\new york\segmentation\new york\256_19_8\process"
    output_folder = r"D:\data\outputdir\new york\segmentation\new york\256_19_8\differences3"

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
            "color_target": TARGET_GREEN
        },
        "images": [],
        "summary": {}
    }

    # 批量处理
    image_results = []
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

            # 计算颜色掩码
            gt_mask, pred_mask = calculate_color_difference(
                img1, img2, TARGET_GREEN, color_tolerance=20
            )

            # 计算语义分割指标
            metrics = calculate_segmentation_metrics(gt_mask, pred_mask)

            # 添加图片信息
            metrics['filename'] = filename
            metrics['image_size'] = f"{img1.shape[1]}x{img1.shape[0]}"
            image_results.append(metrics)

            # 创建可视化图像
            # FN可视化（黄色）：真实道路但预测不是
            fn_visualization = create_segmentation_visualization(gt_mask, pred_mask, metrics, 'FN')

            # FP可视化（红色）：预测道路但真实不是
            fp_visualization = create_segmentation_visualization(gt_mask, pred_mask, metrics, 'FP')

            # 将背景变为透明
            transparent_fn = make_background_transparent(fn_visualization,
                                                         ((fn_visualization[:, :, 2] > 0) |
                                                          (fn_visualization[:, :, 1] > 0)).astype(np.uint8) * 255)

            transparent_fp = make_background_transparent(fp_visualization,
                                                         (fp_visualization[:, :, 2] > 0).astype(np.uint8) * 255)

            # 保存图片
            base_name = os.path.splitext(filename)[0]

            # FN图像
            fn_output_path = os.path.join(output_folder, f"{base_name}_FN.png")
            cv2.imwrite(fn_output_path, transparent_fn)

            # FP图像
            fp_output_path = os.path.join(output_folder, f"{base_name}_FP.png")
            cv2.imwrite(fp_output_path, transparent_fp)

            print(f"  TP: {metrics['TP']} pixels ({metrics.get('TP_percentage', 0):.2f}%)")
            print(f"  FP: {metrics['FP']} pixels ({metrics.get('FP_percentage', 0):.2f}%)")
            print(f"  FN: {metrics['FN']} pixels ({metrics.get('FN_percentage', 0):.2f}%)")
            print(f"  Precision: {metrics.get('Precision', 0):.4f}")
            print(f"  Recall: {metrics.get('Recall', 0):.4f}")
            print(f"  F1 Score: {metrics.get('F1_Score', 0):.4f}")

        except Exception as e:
            print(f"  处理失败: {str(e)}")

    # 存储所有图片的指标
    all_metrics["images"] = image_results

    # 计算总体统计
    if image_results:
        total_tp = sum(m['TP'] for m in image_results)
        total_fp = sum(m['FP'] for m in image_results)
        total_fn = sum(m['FN'] for m in image_results)
        total_tn = sum(m['TN'] for m in image_results)
        total_pixels = sum(m['Total_pixels'] for m in image_results)

        # 计算总体指标
        all_metrics["summary"] = {
            "total_images": len(image_results),
            "average_TP": total_tp / len(image_results),
            "average_FP": total_fp / len(image_results),
            "average_FN": total_fn / len(image_results),
            "average_TN": total_tn / len(image_results),
            "total_TP": total_tp,
            "total_FP": total_fp,
            "total_FN": total_fn,
            "total_TN": total_tn,
            "average_Precision": np.mean([m.get('Precision', 0) for m in image_results]),
            "average_Recall": np.mean([m.get('Recall', 0) for m in image_results]),
            "average_F1_Score": np.mean([m.get('F1_Score', 0) for m in image_results])
        }

        # 保存JSON文件
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)

        print(f"\n处理完成！共处理 {len(image_results)} 对图片")
        print(f"JSON文件已保存到: {json_output_path}")
        print(f"平均指标:")
        print(f"  Precision: {all_metrics['summary']['average_Precision']:.4f}")
        print(f"  Recall: {all_metrics['summary']['average_Recall']:.4f}")
        print(f"  F1 Score: {all_metrics['summary']['average_F1_Score']:.4f}")


# 运行主函数
if __name__ == "__main__":
    main()