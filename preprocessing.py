"""Preprocessing."""

import os
from torchvision import models
from image_processing import segment

src_directory = "ASL_data/"
dst_directory = "processed_data/"

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
          'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
          'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
          'Y', 'Z']

# create the directories
if not os.path.exists(dst_directory):
    os.makedirs(dst_directory)

for i in labels:
    if not os.path.exists(dst_directory + i):
        os.makedirs(dst_directory + i)

print("Downloading model...", end="")
dlab = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True).eval()
print("model downloaded!")

print("\nStarting the foreground background processing...")
for letter in labels:
    for img in os.listdir(src_directory + letter):
        src_path = os.path.join(src_directory+letter, img)
        dst_path = os.path.join(dst_directory+letter, img)

        segment(dlab, src_path, dst_path)
    print(letter, 'FINISHED.')
