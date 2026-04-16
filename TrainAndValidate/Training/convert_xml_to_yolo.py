import os
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import shutil

XML_DIR = "dataset/Annotations"
IMG_DIR = "dataset/JPEGImages"

OUT = "dataset_yolo"

os.makedirs(f"{OUT}/images/train", exist_ok=True)
os.makedirs(f"{OUT}/images/val", exist_ok=True)
os.makedirs(f"{OUT}/labels/train", exist_ok=True)
os.makedirs(f"{OUT}/labels/val", exist_ok=True)

def convert(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.find("filename").text
    w = float(root.find("size/width").text)
    h = float(root.find("size/height").text)

    labels = []

    for obj in root.findall("object"):
        bbox = obj.find("bndbox")

        x1 = float(bbox.find("xmin").text)
        y1 = float(bbox.find("ymin").text)
        x2 = float(bbox.find("xmax").text)
        y2 = float(bbox.find("ymax").text)

        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h

        labels.append(f"0 {cx} {cy} {bw} {bh}")

    return filename, labels


xmls = [f for f in os.listdir(XML_DIR) if f.endswith(".xml")]
train, val = train_test_split(xmls, test_size=0.2, random_state=42)

for split, files in [("train", train), ("val", val)]:
    for xml in files:
        img_name, labels = convert(os.path.join(XML_DIR, xml))

        shutil.copy(
            os.path.join(IMG_DIR, img_name),
            f"{OUT}/images/{split}/{img_name}"
        )

        with open(f"{OUT}/labels/{split}/{img_name.replace('.png','.txt')}", "w") as f:
            f.write("\n".join(labels))
