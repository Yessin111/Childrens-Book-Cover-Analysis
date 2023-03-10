import os
import nltk
from research import VA_r, OD_r, IS_r
from analysis import VA_a, OD_a, IS_a

root = os.getcwd()
n_ages = 2
num_topics = 10
input_path = "/data/book_data.json"
output_path = "/data/book_result.json"
stat_res_path = "data/stat_result.txt"

nltk.download('stopwords')
nltk.download('punkt')

VA_r.get(root, input_path, output_path)
OD_r.get(root, output_path)
IS_r.get(root, output_path)

VA_res = VA_a.analyze(root, output_path, n_ages)
OD_res = OD_a.analyze(root, output_path, n_ages)
IS_res = IS_a.analyze(root, output_path, n_ages, num_topics)

res = {}
res["VA_res"] = VA_res
res["OD_res"] = OD_res
res["IS_res"] = IS_res

with open(stat_res_path, 'wt') as data:
    data.write(str(res))
