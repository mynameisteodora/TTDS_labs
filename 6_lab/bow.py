import os
import re
from chardet import detect


# get file encoding type
def get_encoding_type(file):
    with open(file, 'rb') as f:
        rawdata = f.read()
    return detect(rawdata)['encoding']


def read_tweets(file):
    encoding = get_encoding_type(file)
    f = open(file, 'r', encoding=encoding, errors='ignore')
    lines = f.readlines()

    for i in range(len(lines)):
        lines[i] = lines[i].strip().split('\t')

    return lines


def read_features_bow(features_bow_file):
    features = {}

    with open(features_bow_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if len(line) > 0:
            word, id = line.split('\t')
            features[word.strip()] = id.strip()

    return features


def read_class_ids(class_ids_file):
    class_ids = {}

    with open(class_ids_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if len(line) > 0:
            class_name, class_id = line.split('\t')
            class_ids[class_name.strip()] = class_id.strip()

    return class_ids


def remove_links(tweet):
    tweet = re.sub(r'http\S+', ' ', tweet)

    return tweet


def tokenise(tweet):
    tweet_words = re.sub(r"[^\w\s]|_", " ", tweet).split()

    return tweet_words


def preprocess_tweet(tweet):
    # receives the actual tweet
    tweet = remove_links(tweet).lower()

    tweet_words = tokenise(tweet)

    return tweet_words


def build_feature_dict(*tweet_collections):
    bow = set()
    for tweet_collection in tweet_collections:
        for tweet_info in tweet_collection:
            if len(tweet_info) == 3:
                tweet_id, tweet, category = tweet_info[0], tweet_info[1], tweet_info[2]
                bow.update(preprocess_tweet(tweet))

    bow = [0] + list(bow)
    f = open('feats.bow', 'w')
    for i in range(1, len(bow)):
        f.write('{0}\t {1}\n'.format(bow[i], i))

    return bow


def build_feature_file(original_file, features_bow_file, class_ids_file, destination_file):
    tweet_collection = read_tweets(original_file)
    features_bow = read_features_bow(features_bow_file)
    class_ids = read_class_ids(class_ids_file)

    f = open(destination_file, 'w')

    for tweet_info in tweet_collection:
        if len(tweet_info) == 3:
            tweet_id, tweet, category = tweet_info[0], tweet_info[1], tweet_info[2]
            tweet_words = preprocess_tweet(tweet)

            bow_mapping = {}

            for word in tweet_words:
                condition = word in features_bow.keys()
                if not condition:
                    continue
                else:
                    word_id = features_bow[word]

                    if word_id in bow_mapping.keys():
                        bow_mapping[word_id] += 1
                    else:
                        bow_mapping[word_id] = 1


            #print("Bow mapping = {0}".format(bow_mapping))

            # prepare for printing
            to_print = ""

            for word in tweet_words:
                if word in features_bow.keys():
                    word_id = features_bow[word]
                    #print("word_id = {0}".format(word_id))
                    word_count = bow_mapping[word_id]
                    #print(word_count)
                    to_print += str(word_id) + ':' + str(word_count) + " "

                   # print(to_print)

            category_id = class_ids[category]

            f.write('{0} {1} #{2}\n'.format(category_id, to_print, tweet_id))



if __name__ == '__main__':
    tweet_collection_train = read_tweets('./tweetsclassification/Tweets.14cat.train')
    tweet_collection_test = read_tweets('./tweetsclassification/Tweets.14cat.test')

    # careful! bow is built using only the training set
    bow = build_feature_dict(tweet_collection_train)
    print(bow)

    build_feature_file('./tweetsclassification/Tweets.14cat.train', './feats.bow', './class_id.txt', 'feats.train')
    build_feature_file('./tweetsclassification/Tweets.14cat.test', './feats.bow', './class_id.txt', 'feats.test')
