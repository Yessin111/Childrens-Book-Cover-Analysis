import json
from analysis import ST


def analyze(root, output_path):
    count = [0, 0, 0, 0, 0]
    colors = [{}, {}, {}, {}, {}]
    brightness = [[], [], [], [], []]
    colorfulness = [[], [], [], [], []]
    contrast = [[], [], [], [], []]
    entropy = [[], [], [], [], []]

    with open(root + output_path, 'r') as file_json:
        json_data = json.load(file_json)

        for row in json_data:
            for i in range(5):
                if i in row["age"]:
                    count[i] = count[i] + 1
                    colors[i][row["VA"]["dominant_color_name"]] = colors[i].get(row["VA"]["dominant_color_name"], 0) + 1
                    brightness[i].append(row["VA"]["brightness"])
                    colorfulness[i].append(row["VA"]["colorfulness"]/200)
                    contrast[i].append(row["VA"]["contrast"] / 255)
                    entropy[i].append(row["VA"]["entropy"]/8)

    for i in range(5):
        colors[i] = sorted(colors[i].items(), key=lambda x: x[1], reverse=True)

    res = {}
    res["stats_brightness"] = ST.stat_test(brightness)
    res["stats_colorfulness"] = ST.stat_test(colorfulness)
    res["stats_contrast"] = ST.stat_test(contrast)
    res["stats_entropy"] = ST.stat_test(entropy)

    return res
