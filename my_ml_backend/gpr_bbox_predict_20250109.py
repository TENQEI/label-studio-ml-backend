import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

# transformers 相关
from transformers import (
    AutoImageProcessor,
    PreTrainedModel,
    ViTConfig,
    ViTModel
)


####################################
# 1) 与训练一致的 BBox 回归器
####################################
class BBoxRegressor(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # 特征提取层 (加深网络结构)
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # 中心点预测
        self.center_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 2),
            nn.Sigmoid()
        )

        # 尺寸预测
        self.size_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.feature_extractor(x)

        centers = self.center_predictor(features)  # (cx, cy)
        sizes = self.size_predictor(features)  # (w, h)

        x1 = centers[:, 0] - sizes[:, 0] / 2
        y1 = centers[:, 1] - sizes[:, 1] / 2
        x2 = centers[:, 0] + sizes[:, 0] / 2
        y2 = centers[:, 1] + sizes[:, 1] / 2

        x1 = torch.clamp(x1, 0, 1)
        y1 = torch.clamp(y1, 0, 1)
        x2 = torch.clamp(x2, 0, 1)
        y2 = torch.clamp(y2, 0, 1)

        x2 = torch.max(x2, x1 + 1e-4)
        y2 = torch.max(y2, y1 + 1e-4)

        return torch.stack([x1, y1, x2, y2], dim=1)


####################################
# 2) 与训练一致的多任务模型类
####################################
class MultiTaskViTWithBBox(PreTrainedModel):
    config_class = ViTConfig

    def __init__(self, config, num_main_classes, num_binary_classes=2):
        """
        与训练脚本中保持一致，仅去掉了计算 loss 的部分。
        """
        super().__init__(config)
        self.config = config
        self.config.output_hidden_states = False

        # 1) ViT 主干
        self.vit = ViTModel(self.config)
        hidden_size = config.hidden_size

        # 2) 共享层
        self.shared_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        # 3) 主分类头
        self.main_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_main_classes)
        )

        # 4) 二分类头
        self.binary_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_binary_classes)
        )

        # 5) BBox 回归头（基于上方的 BBoxRegressor）
        self.bbox_regressor = BBoxRegressor(hidden_size)

    def forward(self, pixel_values, return_dict=True):
        # 1) ViT 前向传播
        outputs = self.vit(pixel_values=pixel_values, return_dict=True)
        # 2) 拿到 [CLS] token 特征
        cls_token = outputs.last_hidden_state[:, 0, :]

        # 3) 共享层
        shared_features = self.shared_layer(cls_token)

        # 4) 三个任务的输出
        main_logits = self.main_classifier(shared_features)
        binary_logits = self.binary_classifier(shared_features)
        bbox_pred = self.bbox_regressor(shared_features)

        if not return_dict:
            return main_logits, binary_logits, bbox_pred

        return {
            "main_logits": main_logits,
            "binary_logits": binary_logits,
            "bbox_pred": bbox_pred
        }


####################################
# 3) 工具函数：坐标转换与可视化
####################################
def convert_normalized_to_pascal_voc(bbox, image_width, image_height):
    """
    将 [x1, y1, x2, y2] (0~1 归一化坐标)
    转换为 [x_min, y_min, x_max, y_max] (像素坐标)
    """
    x1, y1, x2, y2 = bbox
    return [
        int(x1 * image_width),
        int(y1 * image_height),
        int(x2 * image_width),
        int(y2 * image_height)
    ]


def draw_bbox(image, bbox, color=(0, 255, 0), thickness=2):
    """
    在图像上绘制边界框
    :param image: 原始图像 (OpenCV格式)
    :param bbox: [x_min, y_min, x_max, y_max] 像素坐标
    """
    x_min, y_min, x_max, y_max = map(int, bbox)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
    return image


####################################
# 4) 推理（Inference）类
####################################
class MyInference:
    def __init__(self, model_path, class_name_mapping=None):
        """
        :param model_path: 训练后模型（配置、权重）所在目录
        :param class_name_mapping: 主分类类别名称映射（列表或字典）
        """
        if class_name_mapping is None:
            class_name_mapping = ["hole", "normal", "pipehole", "pipeline", "rebar", "well", "wellhole"]
        self.class_name_mapping = class_name_mapping

        # 1) 加载配置
        self.config = ViTConfig.from_pretrained(model_path)
        self.config.output_hidden_states = False

        # 2) 加载图像处理器 (和训练时相同)
        self.image_processor = AutoImageProcessor.from_pretrained(model_path)

        # 3) 初始化自定义模型
        num_main_classes = len(class_name_mapping)  # 主分类类别数
        num_binary_classes = 2  # 二分类任务类别数
        self.model = MultiTaskViTWithBBox.from_pretrained(
            model_path,
            config=self.config,
            num_main_classes=num_main_classes,
            num_binary_classes=num_binary_classes
        )
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()

        # 4) 定义推理时使用的 transforms（与训练脚本里的 val_transforms 保持一致）
        size = (self.image_processor.size['height'], self.image_processor.size['width'])
        self.val_transforms = A.Compose([
            A.Resize(height=size[0], width=size[1]),
            A.Normalize(mean=self.image_processor.image_mean, std=self.image_processor.image_std),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels'],
            min_area=0,
            min_visibility=0
        ))

    @torch.no_grad()
    def infer(self, image, known_bbox=None, save_visualization=False):
        """
        对单张图片进行推理，返回主分类 logits、二分类 logits、bbox 预测（归一化坐标）等。
        :param image: 待推理的图像路径/矩阵
        :param known_bbox: 若无目标检测环节，可传入整图 [0,0,1,1]，或其他先验 bbox
        :param save_visualization: 是否保存带预测框的可视化结果
        """
        # 1) 打开图像并转为 numpy
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
            image_np = np.array(image)
        else:
            image_np = image
        h, w = image_np.shape[:2]

        # 2) 若没有给定 bbox，就用整张图作为占位
        if known_bbox is None:
            known_bbox = [0.0, 0.0, 1.0, 1.0]
        pascal_bbox = convert_normalized_to_pascal_voc(known_bbox, w, h)

        # 3) 数据增/预处理
        transformed = self.val_transforms(
            image=image_np,
            bboxes=[pascal_bbox],
            labels=["bbox"]  # label_fields 必需，否则 albumentations 会报错
        )
        pixel_values = transformed["image"].unsqueeze(0)  # (1, C, H, W)

        # 4) 推理
        if torch.cuda.is_available():
            pixel_values = pixel_values.cuda()

        outputs = self.model(pixel_values=pixel_values)
        main_logits = outputs["main_logits"][0].cpu().numpy()  # shape: (num_main_classes,)
        binary_logits = outputs["binary_logits"][0].cpu().numpy()  # shape: (2,)
        bbox_pred = outputs["bbox_pred"][0].cpu().numpy()  # 归一化后的 [x1,y1,x2,y2]

        # 5) 可视化（仅在 save_visualization=True 时执行）
        image_with_bbox = None
        if save_visualization:
            visual_bbox = convert_normalized_to_pascal_voc(bbox_pred, w, h)
            image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            image_with_bbox = draw_bbox(image_cv2.copy(), visual_bbox)

        return main_logits, binary_logits, bbox_pred, image_with_bbox


####################################
# 5) 单图片处理函数
####################################
def process_single_image(
        infer_engine,
        image_path,
        output_dir,
        save_details=False,
        save_visualization=False
):
    """
    处理单张图片并返回结果
    :param infer_engine: MyInference实例
    :param image_path: 图片路径
    :param output_dir: 输出目录，已在外部创建好
    :param save_details: 是否输出详细判识信息txt文件（默认False）
    :param save_visualization: 是否输出可视化绘制后的图像（默认False）
    :return: dict 包含推理结果
    """
    # 生成可视化输出文件名
    image_name = Path(image_path).stem
    output_vis_path = output_dir / f"{image_name}_with_bbox.jpg"

    # 执行推理
    main_logits, binary_logits, bbox_pred, visualized_image = infer_engine.infer(
        str(image_path),
        known_bbox=None,
        save_visualization=save_visualization
    )

    # 若需要保存可视化结果
    if save_visualization and visualized_image is not None:
        cv2.imwrite(str(output_vis_path), visualized_image)
    else:
        output_vis_path = None

    # 解析主分类结果
    main_probs = F.softmax(torch.tensor(main_logits), dim=0).numpy()
    main_pred_idx = np.argmax(main_probs)
    main_pred_label = infer_engine.class_name_mapping[main_pred_idx]
    main_confidence = main_probs[main_pred_idx]

    # 解析二分类结果
    binary_probs = F.softmax(torch.tensor(binary_logits), dim=0).numpy()
    binary_pred_idx = np.argmax(binary_probs)
    binary_confidence = binary_probs[binary_pred_idx]

    # 若需要保存详细信息到文本文件
    detail_path = None
    if save_details:
        detail_info = (
                f"图片分析结果:\n"
                f"1. 主分类结果:\n"
                f"   - 预测类别: {main_pred_label}\n"
                f"   - 置信度: {main_confidence:.4f}\n"
                f"   - 所有类别概率:\n"
                + "\n".join([f"     {cls}: {prob:.4f}"
                             for cls, prob in zip(infer_engine.class_name_mapping, main_probs)])
                + f"\n\n2. 二分类结果:\n"
                  f"   - 预测类别: {binary_pred_idx}\n"
                  f"   - 置信度: {binary_confidence:.4f}\n"
                  f"\n3. 边界框坐标 (归一化):\n"
                  f"   - [x1, y1, x2, y2]: {[f'{x:.4f}' for x in bbox_pred]}"
        )
        detail_path = output_dir / f"{image_name}_details.txt"
        with open(detail_path, 'w', encoding='utf-8') as f:
            f.write(detail_info)

    # 构建结果字典
    result = {
        'image_path': str(image_path),
        'visualization_path': str(output_vis_path) if output_vis_path else None,
        'details_path': str(detail_path) if detail_path else None,
        'main_classification': {
            'predicted_label': main_pred_label,
            'index': int(main_pred_idx),
            'confidence': float(main_confidence),
            'probabilities': main_probs.tolist(),
            'logits': main_logits.tolist()
        },
        'binary_classification': {
            'predicted_class': int(binary_pred_idx),
            'confidence': float(binary_confidence),
            'probabilities': binary_probs.tolist(),
            'logits': binary_logits.tolist()
        },
        'bbox': bbox_pred.tolist()
    }

    return result


def predict_image(infer_engine, image_array):
    """
    处理单张图片并返回结果
    :param infer_engine: MyInference实例
    :param image_array: 图片矩阵

    :return: dict 包含推理结果
    """

    # 执行推理
    main_logits, binary_logits, bbox_pred, _ = infer_engine.infer(image_array)

    # 解析主分类结果
    main_probs = F.softmax(torch.tensor(main_logits), dim=0).numpy()
    main_pred_idx = np.argmax(main_probs)
    main_pred_label = infer_engine.class_name_mapping[main_pred_idx]
    main_confidence = main_probs[main_pred_idx]

    # 解析二分类结果
    binary_probs = F.softmax(torch.tensor(binary_logits), dim=0).numpy()
    binary_pred_idx = np.argmax(binary_probs)
    binary_confidence = binary_probs[binary_pred_idx]

    # 构建结果字典
    result = {
        # 'image_path': str(image_path),
        # 'visualization_path': str(output_vis_path) if output_vis_path else None,
        # 'details_path': str(detail_path) if detail_path else None,
        'main_classification': {
            'predicted_label': main_pred_label,
            'index': int(main_pred_idx),
            'confidence': float(main_confidence),
            'probabilities': main_probs.tolist(),
            'logits': main_logits.tolist()
        },
        'binary_classification': {
            'predicted_class': int(binary_pred_idx),
            'confidence': float(binary_confidence),
            'probabilities': binary_probs.tolist(),
            'logits': binary_logits.tolist()
        },
        'bbox': bbox_pred.tolist()
    }

    return result


####################################
# 6) 文件夹处理函数（最终输出为 CSV）
####################################
def process_directory(
        infer_engine,
        input_dir,
        save_details=False,
        save_visualization=False
):
    """
    处理文件夹中的所有图片，最终结果以CSV格式输出。
    结果保存在 input_dir 下的 results 文件夹（如无则创建）
    :param infer_engine: MyInference实例
    :param input_dir: 输入文件夹路径
    :param save_details: 是否输出每张图片的详细txt文件
    :param save_visualization: 是否输出可视化绘制后的图像
    :return: list of dict 所有图片的推理结果
    """
    input_dir = Path(input_dir)
    # 在 input_dir 下创建一个 results 文件夹
    output_dir = input_dir / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有图片文件
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in input_dir.rglob('*') if f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"No images found in {input_dir}")
        return []

    # 处理所有图片，收集结果
    results = []
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            result = process_single_image(
                infer_engine,
                image_path,
                output_dir=output_dir,
                save_details=save_details,
                save_visualization=save_visualization
            )
            results.append(result)
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue

    # === 将最终结果写入 CSV 文件 ===
    if results:
        csv_path = output_dir / 'analysis_results.csv'

        # 1) 构建表头
        fieldnames = ['image_name']

        # 追加主分类概率列 (使用类别名称作为列名)
        for cls_name in infer_engine.class_name_mapping:
            fieldnames.append(cls_name)

        # 追加二分类概率列 (normal_prob, problem_prob)
        fieldnames.append("normal_prob")
        fieldnames.append("problem_prob")

        # 追加bbox列
        fieldnames.append("BBox")

        # 2) 写入 CSV
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for res in results:
                # 仅取文件名，不含路径
                file_name_only = Path(res['image_path']).name

                row_dict = {}
                row_dict['image_name'] = file_name_only

                # 主分类概率
                main_probs = res['main_classification']['probabilities']
                for cls_name, prob in zip(infer_engine.class_name_mapping, main_probs):
                    row_dict[cls_name] = f"{prob:.4f}"

                # 二分类概率 (二分类：0->normal_prob, 1->problem_prob)
                binary_probs = res['binary_classification']['probabilities']
                row_dict["normal_prob"] = f"{binary_probs[0]:.4f}"
                row_dict["problem_prob"] = f"{binary_probs[1]:.4f}"

                # BBox
                bbox_str = "[" + ", ".join([f"{x:.4f}" for x in res['bbox']]) + "]"
                row_dict["BBox"] = bbox_str

                writer.writerow(row_dict)

        print(f"推理结果已保存到: {csv_path}")

    return results


####################################
# 7) 主函数（仅处理文件夹）
####################################
def main():
    """
    仅支持文件夹输入，不再处理单图片场景。
    使用方式示例：
      1) 设置 model_path, input_dir
      2) 直接运行 main() 即可
    """
    # 1) 设置路径
    # 模型路径：包含模型配置文件(config.json)和权重文件(pytorch_model.bin)的目录
    model_path = r"model-20250109"

    # 输入路径：仅支持文件夹
    input_dir = r"G:\0-GPR_Automation\TestData\3-data\12-jiansheyilu1\Images"

    # 2) 加载推理对象
    class_name_mapping = ["hole", "normal", "pipehole", "pipeline", "rebar", "well", "wellhole"]
    infer_engine = MyInference(model_path, class_name_mapping)

    # 3) 处理文件夹
    results = process_directory(
        infer_engine,
        input_dir,
        save_details=False,  # 如需输出详情txt，改为True
        save_visualization=False  # 如需输出可视化图像，改为True
    )
    print(f"\n=== 文件夹推理完成 ===")
    print(f"- 处理图片数量: {len(results)}")
    print(f"- 结果已保存至: {Path(input_dir) / 'results' / 'analysis_results.csv'}")


if __name__ == "__main__":
    main()
