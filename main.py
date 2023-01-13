import os
from research import VA_r, OD_r, IS_r
from analysis import VA_a, IS_a

root = os.getcwd()
input_path = "/data/book_data.json"
output_path = "/data/result_data.json"

VA_r.get(root, input_path, output_path)
OD_r.get(root, output_path)
IS_r.get(root, output_path)

VA_res = VA_a.analyze(root, output_path)
IS_res = IS_a.analyze(root, output_path)

