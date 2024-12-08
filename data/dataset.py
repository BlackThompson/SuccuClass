import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import torch


class PlantDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # 读取CSV文件
        self.data = pd.read_csv(csv_file)

        # 对species进行编码
        self.label_encoder = LabelEncoder()
        self.data["label"] = self.label_encoder.fit_transform(self.data["species"])

        # Add these lines after the existing label_encoder
        self.family_encoder = LabelEncoder()
        self.genus_encoder = LabelEncoder()
        self.species_encoder = LabelEncoder()

        # Fit the new encoders
        self.data["family_label"] = self.family_encoder.fit_transform(
            self.data["family"]
        )
        self.data["genus_label"] = self.genus_encoder.fit_transform(self.data["genus"])
        self.data["species_label"] = self.species_encoder.fit_transform(
            self.data["species"]
        )

        # 构建图片路径
        self.image_paths = []
        self.labels = []

        for _, row in self.data.iterrows():
            species_dir = os.path.join(
                self.root_dir, row["family"], row["genus"], row["species"]
            )
            if os.path.exists(species_dir):
                for img_name in os.listdir(species_dir):
                    if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.image_paths.append(os.path.join(species_dir, img_name))
                        self.labels.append(row["label"])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        # Get the row index that corresponds to this image
        label_idx = self.labels[idx]
        row = self.data[self.data["label"] == label_idx].iloc[0]

        # Get all classification level labels
        family_label = row["family_label"]
        genus_label = row["genus_label"]
        species_label = row["species_label"]

        # Convert labels to tensors
        family_label = torch.tensor(family_label, dtype=torch.long)
        genus_label = torch.tensor(genus_label, dtype=torch.long)
        species_label = torch.tensor(species_label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        # Instead of returning all labels, return only the species label
        # (or whichever classification level you want to focus on)
        return image, species_label

        # Alternatively, if you want to keep all labels, return them separately:
        # return image, (family_label, genus_label, species_label)

    def get_num_classes(self):
        return len(self.label_encoder.classes_)
