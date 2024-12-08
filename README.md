# SuccuClass: Succulent Plant Classification Using Lightweight Model

![License](https://img.shields.io/badge/license-MIT-brightgreen)

## Usage

```
git clone https://github.com/BlackThompson/SuccuClass.git

pip install -r requirements.txt

python train.py
```

## Explanation

- `spider/` includes the code for  scraping succulent plant images from Bing Images.
- `data/` includes the code to preprocess downloaded datasets.
- `utils/` includes code for train and evaluation modules.
- `data_augmentation.py` is the code for image augmentation.
- `train.py` is the code for training.

## Dataset

We use three datasets in our study, two public datasets [PlantNet-300K](https://github.com/plantnet/PlantNet-300K) and [PlantCLEF](https://www.imageclef.org/) ; and our collocted dataset SuccuClass, which can be downloaded from [Google Drive](https://drive.google.com/drive/folders/11Gz_HvlQgRVem2t5YKlvNRKk-4FocJ7N?usp=drive_link).