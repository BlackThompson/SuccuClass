import os
import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm
import shutil


def parse_xml_files(xml_dir):
    """解析XML文件并创建数据集信息"""
    data = []

    # 遍历所有XML文件
    for xml_file in tqdm(os.listdir(xml_dir), desc="Parsing XML files"):
        if not xml_file.endswith(".xml"):
            continue

        # 获取对应的图片ID (去掉.xml后缀)
        image_id = os.path.splitext(xml_file)[0]

        # 解析XML
        tree = ET.parse(os.path.join(xml_dir, xml_file))
        root = tree.getroot()

        try:
            # 提取需要的信息
            entry = {
                "image_id": image_id,
                "family": root.find("Family").text,
                "genus": root.find("Genus").text,
                "species": root.find("Species").text,
                "content": root.find("Content").text,
                "vote": float(root.find("Vote").text),
                "learn_tag": root.find("LearnTag").text,
            }
            data.append(entry)
        except Exception as e:
            print(f"Error parsing {xml_file}: {str(e)}")
            continue

    return pd.DataFrame(data)


def organize_dataset(xml_dir, output_dir):
    """组织数据集结构"""
    # 解析所有XML文件
    df = parse_xml_files(xml_dir)

    # 只保留训练数据
    df = df[df["learn_tag"] == "Train"]

    # 保存分类信息到CSV
    df.to_csv(os.path.join(output_dir, "classifications.csv"), index=False)

    # 创建目录结构并复制图片
    for _, row in tqdm(df.iterrows(), desc="Organizing images"):
        # 构建目标路径
        dest_path = os.path.join(
            output_dir, "images", row["family"], row["genus"], row["species"]
        )
        os.makedirs(dest_path, exist_ok=True)

        # 复制图片
        src_image = os.path.join(xml_dir, f"{row['image_id']}.jpg")
        if os.path.exists(src_image):
            dest_image = os.path.join(dest_path, f"{row['image_id']}.jpg")
            shutil.copy2(src_image, dest_image)
        else:
            print(f"Warning: Image not found: {src_image}")


if __name__ == "__main__":
    # 设置输入输出路径
    xml_dir = "PlantCLEF"  # 包含XML文件和jpg文件的目录
    output_dir = "dataset"  # 输出目录

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 组织数据集
    organize_dataset(xml_dir, output_dir)

    print("Dataset organization completed!")
