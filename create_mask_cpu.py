import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# -------------------- 路径配置 --------------------
IMAGE_DIR = "images"        # 输入图像文件夹
SAVE_MASK_DIR = "masks"      # 输出黑白mask
SAVE_VIS_DIR = "vis"         # 输出可视化
SAM_MODEL_PATH = "weights/sam_vit_h_4b8939.pth"

os.makedirs(SAVE_MASK_DIR, exist_ok=True)
os.makedirs(SAVE_VIS_DIR, exist_ok=True)


def keep_largest_mask(mask):
    """保留最大前景连通域"""
    labels, num = cv2.connectedComponents(mask.astype(np.uint8))
    if num <= 1:
        return mask

    max_area = 0
    max_label = 0
    for label in range(1, num):
        area = (labels == label).sum()
        if area > max_area:
            max_area = area
            max_label = label

    return (labels == max_label).astype(np.uint8)


def main():
    # -------------------- 始终使用 CPU --------------------
    device = "cpu"
    print("Using CPU for SAM segmentation.")

    # 加载 SAM
    sam = sam_model_registry["vit_h"](checkpoint=SAM_MODEL_PATH)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.9,
        stability_score_thresh=0.95,
        min_mask_region_area=150  # 小碎片过滤
    )

    # -------------------- 遍历图像 --------------------
    for name in os.listdir(IMAGE_DIR):
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(IMAGE_DIR, name)
        print(f"\nProcessing: {img_path}")

        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 生成 mask
        masks = mask_generator.generate(image_rgb)

        if len(masks) == 0:
            print("❌ No mask found.")
            continue

        # 取最大面积的mask
        masks = sorted(masks, key=lambda x: x["area"], reverse=True)
        mask = masks[0]["segmentation"].astype(np.uint8)

        mask = keep_largest_mask(mask)

        # 保存 mask PNG
        mask_output_path = os.path.join(SAVE_MASK_DIR, name.replace(".jpg", ".png"))
        cv2.imwrite(mask_output_path, mask * 255)

        # 保存可视化叠加
        vis = image.copy()
        vis[mask == 1] = (0, 255, 0)  # 绿色
        cv2.imwrite(os.path.join(SAVE_VIS_DIR, name), vis)

        print(f"Saved mask → {mask_output_path}")

    print("\n✨ Done!")


if __name__ == "__main__":
    main()
