import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import spacy
from analysis import ST


def get_name_entity(data, count, n_ages):
    nlp = spacy.load("en_core_web_sm")
    labels = list(nlp.get_pipe("ner").labels)
    animals = ["aardvark", "albatross", "alligator", "alpaca", "ant", "anteater", "antelope", "ape", "armadillo", "donkey", "baboon", "badger", "barracuda", "bat", "bear", "beaver", "bee", "bison", "boar", "buffalo", "butterfly", "camel", "capybara", "caribou", "cassowary", "cat", "caterpillar", "cattle", "chamois", "cheetah", "chicken", "chimpanzee", "chinchilla", "chough", "clam", "cobra", "cockroach", "cod", "cormorant", "coyote", "crab", "crane", "crocodile", "crow", "curlew", "deer", "dinosaur", "dog", "dogfish", "dolphin", "dotterel", "dove", "dragonfly", "duck", "dugong", "dunlin", "eagle", "echidna", "eel", "eland", "elephant", "elk", "emu", "falcon", "ferret", "finch", "fish", "flamingo", "fly", "fox", "frog", "gaur", "gazelle", "gerbil", "giraffe", "gnat", "gnu", "goat", "goldfinch", "goldfish", "goose", "gorilla", "goshawk", "grasshopper", "grouse", "guanaco", "gull", "hamster", "hare", "hawk", "hedgehog", "heron", "herring", "hippopotamus", "hornet", "horse", "human", "hummingbird", "hyena", "ibex", "ibis", "jackal", "jaguar", "jay", "jellyfish", "kangaroo", "kingfisher", "koala", "kookabura", "kouprey", "kudu", "lapwing", "lark", "lemur", "leopard", "lion", "llama", "lobster", "locust", "loris", "louse", "lyrebird", "magpie", "mallard", "manatee", "mandrill", "mantis", "marten", "meerkat", "mink", "mole", "mongoose", "monkey", "moose", "mosquito", "mouse", "mule", "narwhal", "newt", "nightingale", "octopus", "okapi", "opossum", "oryx", "ostrich", "otter", "owl", "oyster", "panther", "parrot", "partridge", "peafowl", "pelican", "penguin", "pheasant", "pig", "pigeon", "pony", "porcupine", "porpoise", "quail", "quelea", "quetzal", "rabbit", "raccoon", "rail", "ram", "rat", "raven", "reindeer", "rhinoceros", "rook", "salamander", "salmon", "sandpiper", "sardine", "scorpion", "seahorse", "seal", "shark", "sheep", "shrew", "skunk", "snail", "snake", "sparrow", "spider", "spoonbill", "squid", "squirrel", "starling", "stingray", "stinkbug", "stork", "swallow", "swan", "tapir", "tarsier", "termite", "tiger", "toad", "trout", "turkey", "turtle", "viper", "vulture", "wallaby", "walrus", "wasp", "weasel", "whale", "wildcat", "wolf", "wolverine", "wombat", "woodcock", "woodpecker", "worm", "wren", "yak", "zebra"]

    results = []

    for i in range(n_ages):
        results.append([])
        for j in range(len(labels)+1):
            results[i].append(0)

    for i in range(n_ages):
        for caption in data[i]:
            this = 0
            for animal in animals:
                this = this + caption.lower().count(animal)
            doc = nlp(caption)
            for ent in doc.ents:
                results[i][labels.index(ent.label_) + 1] = results[i][labels.index(ent.label_) + 1] + 1

    for i in range(n_ages):
        total = 0
        for caption in data[i]:
            this = 0
            for animal in animals:
                this = this + caption.lower().count(animal)
            total = total + this
        results[i][0] = total

    return results, ST.chi(results, count, n_ages, 19)


def get_topic_list(data, count, n_ages, num_topics):
    language = 'english'
    stop_words = set(stopwords.words(language))

    tokenized = [[]] * n_ages

    for i in range(n_ages):
        for caption in data[i]:
            words = word_tokenize(caption)
            tokenized[i].append([word for word in words if
                                                 word not in stop_words and word not in [",", "-", "book", "cover",
                                                                                         "illustration", "poster",
                                                                                         "magazine", "picture",
                                                                                         "movie", "novel", "cartoon"] and "'" not in word])

    flat_list = [item for sublist in tokenized for item in sublist]

    dictionary = Dictionary(flat_list)
    corpus = [dictionary.doc2bow(caption) for caption in flat_list]

    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
    topics = lda.get_document_topics(corpus)

    results = [[]] * n_ages

    for i in range(n_ages):
        for j in range(num_topics):
            results[i].append(0)

        for topic in topics:
            most_likely_topic = max(topic, key=lambda x: x[1])[0]
            results[i][most_likely_topic] = results[i][most_likely_topic] + 1

    return results, ST.chi(results, count, n_ages, num_topics)


def analyze(root, output_path, n_ages, num_topics):
    count = [0] * n_ages
    captions = [[]] * n_ages

    with open(root + output_path, 'r') as file_json:
        json_data = json.load(file_json)

        for row in json_data:
            if "IS" in row:
                for i in range(n_ages):
                    if i in row["age"]:
                        count[i] = count[i] + 1
                        captions[i].append(row["IS"])

    res = {}
    res["topic"] = get_topic_list(captions, count, n_ages, num_topics)
    res["ner"] = get_name_entity(captions, count, n_ages)

    return res

