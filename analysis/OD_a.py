import json
import math
import warnings
import pandas as pd
from scipy.stats import pearsonr


def analyze(root, output_path, n_ages):
    labels = ['tortoise', 'magpie', 'sea turtle', 'general football', 'ambulance', 'ladder', 'toothbrush', 'syringe', 'sink', 'toy', 'organ', 'apple', 'eye', 'cosmetics', 'paddle', 'snowman', 'beer', 'chopsticks', 'beard', 'bird', 'traffic light', 'croissant', 'cucumber', 'radish', 'towel', 'doll', 'skull', 'washing machine', 'glove', 'belt', 'sunglasses', 'banjo', 'cart', 'ball', 'backpack', 'bike', 'home appliance', 'centipede', 'boat', 'surfboard', 'boot', 'headphones', 'hot dog', 'shorts', 'fast food', 'bus', 'boy', 'bicycle wheel', 'barge', 'laptop', 'miniskirt', 'drill', 'dress', 'bear', 'waffle', 'pancake', 'brown bear', 'woodpecker', 'blue jay', 'pretzel', 'bagel', 'tower', 'teapot', 'person', 'bow and arrow', 'swimwear', 'beehive', 'brassiere', 'bee', 'bat', 'starfish', 'popcorn', 'burrito', 'chainsaw', 'balloon', 'tent', 'licence plate', 'lantern', 'flashlight', 'billboard', 'tiara', 'limousine', 'necklace', 'carnivore', 'scissors', 'stairs', 'computer keyboard', 'printer', 'traffic sign', 'chair', 'shirt', 'poster', 'cheese', 'sock', 'fire hydrant', 'land vehicle', 'earrings', 'tie', 'watercraft', 'cabinetry', 'suitcase', 'muffin', 'bidet', 'snack', 'snowmobile', 'clock', 'medical equipment', 'cattle', 'cello', 'jet ski', 'camel', 'coat', 'suit', 'desk', 'cat', 'bronze sculpture', 'juice', 'gondola', 'beetle', 'cannon', 'mouse', 'cookie', 'office', 'fountain', 'coin', 'calculator', 'cocktail', 'computer monitor', 'box', 'christmas tree', 'cowboy hat', 'hiking equipment', 'studio couch', 'drum', 'dessert', 'wine rack', 'drink', 'zucchini', 'ladle', 'mouth', 'dairy', 'dice', 'oven', 'dinosaur', 'couch', 'cricket ball', 'winter melon', 'whiteboard', 'door', 'hat', 'shower', 'fedora', 'guacamole', 'dagger', 'scarf', 'dolphin', 'sombrero', 'tin can', 'mug', 'tap', 'harbor seal', 'stretcher', 'goggles', 'human body', 'roller skates', 'coffee cup', 'cutting board', 'blender', 'plumbing fixture', 'stop sign', 'office supplies', 'volleyball', 'vase', 'slow cooker', 'wardrobe', 'coffee', 'paper towel', 'personal care', 'food', 'sun hat', 'tree house', 'skirt', 'gas stove', 'salt and pepper shakers', 'mechanical fan', 'fruit', 'french fries', 'nightstand', 'barrel', 'kite', 'tart', 'treadmill', 'fox', 'flag', 'horn', 'window blind', 'foot', 'golf cart', 'jacket', 'egg', 'street light', 'guitar', 'pillow', 'leg', 'isopod', 'grape', 'ear', 'power plugs and sockets', 'panda', 'giraffe', 'woman', 'door handle', 'rhinoceros', 'bathtub', 'goldfish', 'houseplant', 'goat', 'baseball bat', 'baseball glove', 'mixing bowl', 'marine invertebrates', 'kitchen utensil', 'light switch', 'house', 'horse', 'stationary bicycle', 'ceiling fan', 'sofa bed', 'harp', 'sandal', 'bicycle helmet', 'saucer', 'harpsichord', 'hair', 'hamster', 'curtain', 'bed', 'kettle', 'fireplace', 'scale', 'drinking straw', 'insect', 'invertebrate', 'food processor', 'bookcase', 'refrigerator', 'wood-burning stove', 'punching bag', 'common fig', 'jaguar', 'golf ball', 'fashion accessory', 'alarm clock', 'filing cabinet', 'artichoke', 'table', 'tableware', 'kangaroo', 'koala', 'knife', 'bottle', 'lynx', 'lavender', 'lighthouse', 'dumbbell', 'head', 'bowl', 'porch', 'lizard', 'billiard table', 'mammal', 'mouse', 'motorcycle', 'musical instrument', 'swim cap', 'frying pan', 'snowplow', 'bathroom cabinet', 'missile', 'bust', 'man', 'milk', 'plate', 'mobile phone', 'baked goods', 'mushroom', 'pitcher', 'mirror', 'lifejacket', 'table tennis racket', 'musical keyboard', 'scoreboard', 'briefcase', 'kitchen knife', 'tennis ball', 'plastic bag', 'oboe', 'chest of drawers', 'ostrich', 'piano', 'girl', 'plant', 'potato', 'sports equipment', 'pasta', 'penguin', 'pumpkin', 'pear', 'infant bed', 'polar bear', 'mixer', 'cupboard', 'jacuzzi', 'pizza', 'digital clock', 'pig', 'reptile', 'rifle', 'lipstick', 'skateboard', 'raven', 'high heels', 'red panda', 'rose', 'rabbit', 'sculpture', 'saxophone', 'shotgun', 'seafood', 'submarine sandwich', 'snowboard', 'sword', 'picture frame', 'sushi', 'loveseat', 'ski', 'squirrel', 'tripod', 'stethoscope', 'submarine', 'scorpion', 'segway', 'bench', 'snake', 'coffee table', 'skyscraper', 'sheep', 'television', 'trombone', 'tea', 'tank', 'taco', 'telephone', 'tiger', 'strawberry', 'trumpet', 'tree', 'tomato', 'train', 'tool', 'picnic basket', 'trousers', 'bowling equipment', 'football helmet', 'truck', 'coffeemaker', 'violin', 'vehicle', 'handbag', 'wine', 'weapon', 'wheel', 'worm', 'wok', 'whale', 'zebra', 'auto part', 'jug', 'cream', 'monkey', 'lion', 'bread', 'platter', 'chicken', 'eagle', 'helicopter', 'owl', 'duck', 'turtle', 'hippopotamus', 'crocodile', 'toilet', 'toilet paper', 'squid', 'clothing', 'footwear', 'lemon', 'spider', 'deer', 'frog', 'banana', 'rocket', 'wine glass', 'countertop', 'tablet computer', 'waste container', 'swimming pool', 'dog', 'book', 'elephant', 'shark', 'candle', 'leopard', 'porcupine', 'flower', 'canary', 'cheetah', 'palm tree', 'hamburger', 'maple', 'building', 'fish', 'lobster', 'asparagus', 'furniture', 'hedgehog', 'airplane', 'spoon', 'otter', 'bull', 'oyster', 'convenience store', 'bench', 'ice cream', 'caterpillar', 'butterfly', 'parachute', 'orange', 'antelope', 'moths and butterflies', 'window', 'closet', 'castle', 'jellyfish', 'goose', 'mule', 'swan', 'peach', 'seat belt', 'raccoon', 'fork', 'lamp', 'camera', 'squash', 'racket', 'face', 'arm', 'vegetable', 'unicycle', 'falcon', 'snail', 'shellfish', 'cabbage', 'carrot', 'mango', 'jeans', 'flowerpot', 'pineapple', 'drawer', 'stool', 'envelope', 'cake', 'dragonfly', 'sunflower', 'microwave oven', 'honeycomb', 'marine mammal', 'sea lion', 'ladybug', 'shelf', 'watch', 'candy', 'salad', 'parrot', 'handgun', 'sparrow', 'van', 'spice rack', 'light bulb', 'corded phone', 'sports uniform', 'tennis racket', 'wall clock', 'serving tray', 'kitchen & dining room table', 'dog bed', 'cake stand', 'bathroom accessory', 'kitchen appliance', 'tire', 'ruler', 'luggage and bags', 'microphone', 'broccoli', 'umbrella', 'pastry', 'grapefruit', 'animal', 'bell pepper', 'turkey', 'lily', 'pomegranate', 'doughnut', 'glasses', 'nose', 'pen', 'ant', 'car', 'aircraft', 'hand', 'teddy bear', 'watermelon', 'cantaloupe', 'dishwasher', 'flute', 'balance beam', 'sandwich', 'shrimp', 'sewing machine', 'binoculars', 'rays and skates', 'ipod', 'accordion', 'willow', 'crab', 'crown', 'seahorse', 'perfume', 'alpaca', 'taxi', 'canoe', 'remote control', 'wheelchair', 'rugby ball', 'helmet']
    count = [0] * n_ages
    tuples = []
    objects = [{}] * n_ages
    objects_clean = [[]] * n_ages
    co_dupes = [{}] * n_ages
    co_clean = [{}] * n_ages
    PMI = []

    for i in range(n_ages):
        for el in labels:
            objects[i][el] = 0

    with open(root + output_path, 'r') as file_json:
        json_data = json.load(file_json)

        for row in json_data:
            if "OD" in row:
                for i in range(n_ages):
                    if i in row["age"]:
                        count[i] = count[i] + 1
                        temp = []
                        for el in row["OD"]:
                            if row["OD"][el] >= 0.2:
                                objects[i][el] = objects[i][el] + 1
                                temp.append(el)
                        for el1 in temp:
                            for el2 in temp:
                                if el1 != el2:
                                    co_dupes[i][(el1, el2)] = co_dupes[i].get((el1, el2), 0) + 1

    df = pd.DataFrame({
        'labels': labels
    })

    string = []
    number = []

    for i in range(n_ages):
        for el in labels:
            objects_clean[i].append(objects[i][el])
        df.append({i: objects_clean[i]}, ignore_index=True)
        string.append(str(i))
        number.append(i)

    for label in df['labels']:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                r, p = pearsonr(df[df['labels'] == label][string].values[0], number)
                tuples.append((label, round(r, 2), round(p, 2)))
            except:
                pass

    for i in range(n_ages):
        for el in sorted(co_dupes[i].items(), key=lambda x: x[1], reverse=True):
            (x, y) = el[0]
            if (y, x) not in co_clean[i].keys():
                co_clean[i][(x, y)] = el[1]

    for i in range(n_ages):
        pmi_data = []
        total_occurence = 0

        for el in co_clean[i]:
            total_occurence = total_occurence + co_clean[i][el]
        for el in co_clean[i]:
            x = el[0]
            y = el[1]
            co_occurence = co_clean[i][el]
            p_x = objects[i][x] / total_occurence
            p_y = objects[i][y] / total_occurence
            p_xy = co_occurence / total_occurence
            pmi = round(math.log2(p_xy / (p_x * p_y)), 3)
            pmi_data.append(((x, y), pmi))

        pmi_data = sorted(pmi_data, key=lambda x: x[1], reverse=True)
        PMI.append(pmi_data)

    res = {}
    res["frequency"] = objects
    res["correlation"] = tuples
    res["co-occurence"] = PMI

    return res
