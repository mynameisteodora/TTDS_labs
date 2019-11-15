import os
import re
from chardet import detect
import collections
import numpy as np

num_classes = 14


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
                    word_id = int(features_bow[word])

                    if word_id in bow_mapping.keys():
                        bow_mapping[word_id] += 1
                    else:
                        bow_mapping[word_id] = 1

            # prepare for printing
            to_print = ""

            ordered_bow = collections.OrderedDict(sorted(bow_mapping.items()))
            for bow_word, count in ordered_bow.items():
                to_print += str(bow_word) + ':' + str(count) + " "

            category_id = class_ids[category]

            f.write('{0} {1} #{2}\n'.format(category_id, to_print, tweet_id))


def evaluate_model(features_test_file, predictions_file, output_file):
    f = open(features_test_file, 'r')
    g = open(predictions_file, 'r')
    out = open(output_file, 'w')

    confusion_matrix = np.zeros((num_classes, num_classes))
    test_file_lines = f.readlines()
    true_classes = []

    for i in range(len(test_file_lines)):
        curr_line = test_file_lines[i]
        if curr_line != '':
            true_class = curr_line.split()[0]
            true_classes.append(int(true_class))

    pred_file_lines = g.readlines()
    correct_classes = 0
    total_nb_of_classes = len(true_classes)

    for i in range(len(pred_file_lines)):
        curr_line = pred_file_lines[i]
        if curr_line != '':
            pred_class = int(curr_line.split()[0])
            if pred_class == true_classes[i]:
                correct_classes += 1

            pred_class_idx = pred_class - 1
            correct_class_idx = true_classes[i] - 1

            confusion_matrix[pred_class_idx][correct_class_idx] += 1

    accuracy = correct_classes / total_nb_of_classes
    print("Accuracy = {0}".format(accuracy))

    # compute precision, recall and f1 for each class
    precisions = []
    recalls = []
    f1s = []

    for i in range(num_classes):
        # precision for class i
        correct_is = sum(confusion_matrix[:, i])
        classified_as_i = sum(confusion_matrix[i, :])
        correctly_predicted_i = confusion_matrix[i][i]
        prec = correctly_predicted_i / classified_as_i

        # recall for class i
        recall = correctly_predicted_i / correct_is

        f1 = (2 * prec * recall) / (prec + recall)

        precisions.append(prec)
        recalls.append(recall)
        f1s.append(f1)

    # compute macro-f1
    macro_f1 = sum(f1s) / num_classes

    print("Confusion matrix = \n{0}".format(confusion_matrix))
    print("Accuracy = {0}".format(('%0.3f' % accuracy)))
    print("Macro-f1 = {0}".format('%0.3f' % macro_f1))

    out.write("Accuracy = {0}\n".format(('%0.3f' % accuracy)))
    out.write("Macro-f1 = {0}\n".format('%0.3f' % macro_f1))

    for i in range(num_classes):
        print("{0}: P={1} R={2} F={3}".format(i + 1, '%0.3f' % precisions[i],
                                              '%0.3f' % recalls[i],
                                              '%0.3f' % f1s[i]))

        out.write("{0}: P={1} R={2} F={3}\n".format(i + 1, '%0.3f' % precisions[i],
                                              '%0.3f' % recalls[i],
                                              '%0.3f' % f1s[i]))

    return accuracy


if __name__ == '__main__':
    tweet_collection_train = read_tweets('./tweetsclassification/Tweets.14cat.train')
    tweet_collection_test = read_tweets('./tweetsclassification/Tweets.14cat.test')

    # careful! bow is built using only the training set
    bow = build_feature_dict(tweet_collection_train)
    print(bow)

    build_feature_file('./tweetsclassification/Tweets.14cat.train', './feats.bow', './class_id.txt', 'feats.train')
    build_feature_file('./tweetsclassification/Tweets.14cat.test', './feats.bow', './class_id.txt', 'feats.test')

    print(evaluate_model('./feats.test', './pred.out', 'Eval.txt'))
