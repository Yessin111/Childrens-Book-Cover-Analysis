import json
from analysis import ST


def analyze(root, output_path, n_ages):
    count = [0] * n_ages
    colors = [{}] * n_ages
    brightness = [[]] * n_ages
    colorfulness = [[]] * n_ages
    contrast = [[]] * n_ages
    entropy = [[]] * n_ages

    with open(root + output_path, 'r') as file_json:
        json_data = json.load(file_json)

        for row in json_data:
            if "VA" in row:
                for i in range(n_ages):
                    if i in row["age"]:
                        count[i] = count[i] + 1
                        colors[i][row["VA"]["dominant_color_name"]] = colors[i].get(row["VA"]["dominant_color_name"], 0) + 1
                        brightness[i].append(row["VA"]["brightness"])
                        colorfulness[i].append(row["VA"]["colorfulness"]/200)
                        contrast[i].append(row["VA"]["contrast"] / 255)
                        entropy[i].append(row["VA"]["entropy"]/8)

    for i in range(n_ages):
        colors[i] = sorted(colors[i].items(), key=lambda x: x[1], reverse=True)

    res = {}
    res["stats_brightness"] = ST.anova_turkey(brightness, n_ages)
    res["stats_colorfulness"] = ST.anova_turkey(colorfulness, n_ages)
    res["stats_contrast"] = ST.anova_turkey(contrast, n_ages)
    res["stats_entropy"] = ST.anova_turkey(entropy, n_ages)

    return res
