import os
import shutil
from PIL import Image
import numpy as np
from torchvision import transforms
import torch
from tqdm import tqdm


def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def add_noise(image, noise_factor=0.05):
    img_array = np.array(image)
    noise = np.random.normal(0, noise_factor * 255, img_array.shape)
    noisy_img = img_array + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)


def augment_image(image, index):
    """Perform data augmentation on a single image to generate multiple variants"""
    augmentations = []

    # Basic transformations
    basic_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomResizedCrop(
                size=(image.size[1], image.size[0]), scale=(0.8, 1.0)
            ),
        ]
    )

    # 1. Add noise to original image
    augmentations.append(add_noise(image))

    # 2. Horizontal flip + brightness adjustment
    aug_img = transforms.RandomHorizontalFlip(p=1.0)(image)
    aug_img = transforms.ColorJitter(brightness=0.2)(aug_img)
    augmentations.append(aug_img)

    # 3. Random rotation + contrast adjustment
    aug_img = transforms.RandomRotation(30)(image)
    aug_img = transforms.ColorJitter(contrast=0.2)(aug_img)
    augmentations.append(aug_img)

    # 4. Vertical flip + saturation adjustment
    aug_img = transforms.RandomVerticalFlip(p=1.0)(image)
    aug_img = transforms.ColorJitter(saturation=0.2)(aug_img)
    augmentations.append(aug_img)

    # 5. Random crop and resize
    aug_img = transforms.RandomResizedCrop(
        size=(image.size[1], image.size[0]), scale=(0.8, 1.0)
    )(image)
    augmentations.append(aug_img)

    # 6. Combined transformations
    aug_img = basic_transforms(image)
    augmentations.append(aug_img)

    return augmentations


def augment_dataset(src_root, dst_root):
    """Augment the entire dataset"""
    # 首先复制原始数据集
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
    shutil.copytree(src_root, dst_root)

    # 遍历所有图片文件
    for family in tqdm(os.listdir(src_root), desc="Processing families"):
        family_path = os.path.join(src_root, family)
        if not os.path.isdir(family_path):
            continue

        for genus in os.listdir(family_path):
            genus_path = os.path.join(family_path, genus)
            if not os.path.isdir(genus_path):
                continue

            for species in os.listdir(genus_path):
                species_path = os.path.join(genus_path, species)
                if not os.path.isdir(species_path):
                    continue

                # 创建目标目录
                dst_species_path = os.path.join(dst_root, family, genus, species)
                create_dir_if_not_exists(dst_species_path)

                # 处理每张图片
                for img_name in os.listdir(species_path):
                    if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                        continue

                    img_path = os.path.join(species_path, img_name)
                    try:
                        # 读取原图
                        image = Image.open(img_path).convert("RGB")

                        # 生成增强图片
                        augmented_images = augment_image(image, 0)

                        # 保存增强后的图片
                        name, ext = os.path.splitext(img_name)
                        for i, aug_img in enumerate(augmented_images):
                            aug_name = f"{name}_aug_{i}{ext}"
                            aug_path = os.path.join(dst_species_path, aug_name)
                            aug_img.save(aug_path, quality=95)

                    except Exception as e:
                        print(f"Error processing {img_path}: {str(e)}")


if __name__ == "__main__":
    src_root = "data_English"
    dst_root = "data_augmentation"

    print("Starting data augmentation...")
    augment_dataset(src_root, dst_root)
    print("Data augmentation completed!")

    # 统计文件数量
    original_count = sum([len(files) for r, d, files in os.walk(src_root)])
    augmented_count = sum([len(files) for r, d, files in os.walk(dst_root)])

    print(f"Original dataset size: {original_count} images")
    print(f"Augmented dataset size: {augmented_count} images")
    print(f"Augmentation ratio: {augmented_count/original_count:.2f}x")
