import cv2
import os
import albumentations as A
import numpy as np

dataset_dir = "dataset"
num_augmentations = 5

transform = A.Compose([

    # ✅ REAL ZOOM (Visible)
    A.RandomResizedCrop(
        size=(224, 224),     # NEW SYNTAX
        scale=(0.6, 1.0),    # 0.6 = strong zoom in
        ratio=(0.9, 1.1),
        p=0.9
    ),

    # ✅ Strong Rotation
    A.Rotate(
        limit=35,
        border_mode=cv2.BORDER_REFLECT_101,
        p=0.8
    ),

    # ✅ Flip
    A.HorizontalFlip(p=0.5),

    # ✅ Brightness / Contrast
    A.RandomBrightnessContrast(
        brightness_limit=0.3,
        contrast_limit=0.3,
        p=0.6
    ),

    # ✅ Motion Blur
    A.MotionBlur(blur_limit=5, p=0.4),

    # ✅ Noise
    A.GaussNoise(var_limit=(10.0, 40.0), p=0.4),

    # ✅ Small Occlusion
    A.CoarseDropout(
        max_holes=1,
        max_height=30,
        max_width=30,
        fill_value=0,
        p=0.3
    )

])

for person in os.listdir(dataset_dir):

    person_path = os.path.join(dataset_dir, person)

    if not os.path.isdir(person_path):
        continue

    print(f"Augmenting: {person}")

    for img_name in os.listdir(person_path):

        if img_name.startswith("aug_"):
            continue

        img_path = os.path.join(person_path, img_name)
        image = cv2.imread(img_path)

        if image is None:
            continue

        for i in range(num_augmentations):

            augmented = transform(image=image)["image"]

            if np.std(augmented) < 20:
                continue

            cv2.imwrite(
                os.path.join(person_path, f"aug_{i}_{img_name}"),
                augmented
            )

print("✅ Augmentation Completed Successfully")