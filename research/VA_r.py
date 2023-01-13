import json
import urllib.request
import cv2
import numpy as np
from PIL import Image, ImageStat
from sklearn.cluster import KMeans
import skimage.measure
from data.colors import color_names

VA = []


def get_color_name(rgb_triplet):
    min_colours = {}
    for row in color_names:
        r_c, g_c, b_c = list(row.keys())[0]
        rd = (r_c - rgb_triplet[0]) ** 2
        gd = (g_c - rgb_triplet[1]) ** 2
        bd = (b_c - rgb_triplet[2]) ** 2
        min_colours[(rd + gd + bd)] = list(row.values())[0]
    return min_colours[min(min_colours.keys())]


def get_dominant_color(input_image):
    img = input_image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    kmeans = KMeans(n_clusters=9, n_init="auto")
    kmeans.fit(img)
    color = abs(kmeans.cluster_centers_.round())[np.bincount(kmeans.labels_).argmax()]
    color = (int(color[0]), int(color[1]), int(color[2]))
    color_name = get_color_name(color)
    return color, color_name


def get_brightness(input_image):
    brightness_image = Image.fromarray(input_image).convert("L")
    stat = ImageStat.Stat(brightness_image)
    return round(stat.mean[0] / 255, 3)


def get_colorfulness(input_image):
    (R, G, B) = cv2.split(input_image.astype("float"))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    std_root = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    mean_root = np.sqrt((rbMean ** 2) + (ybMean ** 2))
    # return round((std_root + (0.3 * mean_root)) / 100, 3)
    return round((std_root + (0.3 * mean_root)), 3)


def get_contrast(input_image):
    contrast_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    contrast = contrast_image.std()
    return round(contrast, 3)


def get_entropy(input_image):
    entropy = skimage.measure.shannon_entropy(input_image)
    return round(entropy, 3)


def retrieve(image):
    arr = np.asarray(bytearray(image.read()), dtype=np.uint8)
    image = cv2.imdecode(arr, -1)

    dom_color = get_dominant_color(image)

    entry = {
        "dominant_color_rgb": dom_color[0],
        "dominant_color_name": dom_color[1],
        "brightness": get_brightness(image),
        "colorfulness": get_colorfulness(image),
        "contrast": get_contrast(image),
        "entropy": get_entropy(image)
    }

    return entry


def get(root, input_path, output_path):
    print("Visual attributes")
    with open(root + input_path, "r") as file_json:
        json_data = json.load(file_json)

    for i, row in enumerate(json_data):
        print("Book: " + str(i+1))
        try:
            with urllib.request.urlopen(row["cover"]) as image:
                row["VA"] = retrieve(image)
        except:
            pass
        VA.append(row)

    open(root + output_path, "w")

    with open(root + output_path, "r+") as file_json:
        file_json.seek(0)
        json.dump(VA, file_json, indent=4)
