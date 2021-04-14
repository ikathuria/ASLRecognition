"""Preprocessing."""

import os
from torchvision import models
from image_processing import segment

src_directory = "ASL_data/"
dst_directory = "ASL_data_processed/"

labels = ['B', 'C', 'D', 'E', 'F', 'G', 'H',
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
i = 0
while i < 26:
    for letter in labels:
        i += 1
        for img in os.listdir(src_directory + letter):
            count = {
                "A": len(os.listdir(dst_directory + "A")),
                "B": len(os.listdir(dst_directory + "B")),
                "C": len(os.listdir(dst_directory + "C")),
                "D": len(os.listdir(dst_directory + "D")),
                "E": len(os.listdir(dst_directory + "E")),
                "F": len(os.listdir(dst_directory + "F")),
                "G": len(os.listdir(dst_directory + "G")),
                "H": len(os.listdir(dst_directory + "H")),
                "I": len(os.listdir(dst_directory + "I")),
                "J": len(os.listdir(dst_directory + "J")),
                "K": len(os.listdir(dst_directory + "K")),
                "L": len(os.listdir(dst_directory + "L")),
                "M": len(os.listdir(dst_directory + "M")),
                "N": len(os.listdir(dst_directory + "N")),
                "O": len(os.listdir(dst_directory + "O")),
                "P": len(os.listdir(dst_directory + "P")),
                "Q": len(os.listdir(dst_directory + "Q")),
                "R": len(os.listdir(dst_directory + "R")),
                "S": len(os.listdir(dst_directory + "S")),
                "T": len(os.listdir(dst_directory + "T")),
                "U": len(os.listdir(dst_directory + "U")),
                "V": len(os.listdir(dst_directory + "V")),
                "W": len(os.listdir(dst_directory + "W")),
                "X": len(os.listdir(dst_directory + "X")),
                "Y": len(os.listdir(dst_directory + "Y")),
                "Z": len(os.listdir(dst_directory + "Z")),
            }

            print(letter, count[letter])
            src_path = os.path.join(src_directory+letter, img)
            dst_path = os.path.join(dst_directory+letter, img)

            segment(dlab, src_path, dst_path)

        print(letter, 'FINISHED.')
