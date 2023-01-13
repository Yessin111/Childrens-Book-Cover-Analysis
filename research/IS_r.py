from lavis.models import load_model_and_preprocess
import json
import requests
import torch
from io import BytesIO
from PIL import Image


IS = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="large_coco", is_eval=True, device=device)
vis_processors.keys()


def retrieve(image):
    raw_image = Image.open(BytesIO(image.content)).convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    caption = model.generate({"image": image})
    return caption[0]


def get(root, output_path):
    print("\nImplied Story")
    with open(root + output_path, "r") as file_json:
        json_data = json.load(file_json)

    for i, row in enumerate(json_data):
        print("Book: " + str(i+1))
        try:
            with requests.get(row["cover"]) as image:
                row["IS"] = retrieve(image)
        except:
            pass
        IS.append(row)

    with open(root + output_path, "r+") as file_json:
        file_json.seek(0)
        json.dump(IS, file_json, indent=4)
