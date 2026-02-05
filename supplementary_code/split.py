import os
import shutil
from sklearn.model_selection import train_test_split

image_source_dir = "finetune/walnut/images"  # image dir
label_source_dir = "finetune/walnut/labels"  # labels dir
output_dir = "finetune/walnut"

train_ratio = 0.6  # split ratios
val_ratio = 0.2

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

images = [f for f in os.listdir(image_source_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# split dataset
train, test = train_test_split(images, test_size=1 - train_ratio, random_state=42)
val, test = train_test_split(test, test_size=val_ratio / (1 - train_ratio), random_state=42)


def copy_files(image_files, split):
    for img in image_files:
        shutil.copy(os.path.join(image_source_dir, img), os.path.join(output_dir, "images", split, img))

        label_file = img.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
        label_src = os.path.join(label_source_dir, label_file)
        if os.path.exists(label_src):
            shutil.copy(label_src, os.path.join(output_dir, "labels", split, label_file))


copy_files(train, "train")
copy_files(val, "val")
copy_files(test, "test")

print("Dataset split completed!")