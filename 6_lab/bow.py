import collections
import re

import numpy as np
import requests
from bs4 import BeautifulSoup
from chardet import detect

from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import *
from nltk.corpus import stopwords

num_classes = 14
tokeniser = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
stemmer = PorterStemmer()
stops = set(stopwords.words('english'))

emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
])
emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
])


def retrieve_url(url):
    try:
        resp = requests.head(url, allow_redirects=True, timeout=5)
    except:
        return ""
    # while resp.status_code == 301:
    #     resp = requests.head(resp.headers["Location"])
    if resp.status_code == 200:
        return resp.url
    else:
        return ""


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
    # removes link but puts back the words in the page title of that link

    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweet)

    headers = {
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36'}

    for url in urls:
        og_url = retrieve_url(url)
        if og_url != "":
            try:
                response = requests.get(og_url, headers=headers, timeout=5)
            except:
                continue

            soup = BeautifulSoup(response.text, 'lxml')
            if soup.title:
                link_title = soup.title.text
                tweet += " " + link_title

    tweet = re.sub(r'http\S+', ' ', tweet)

    print("Final tweet = {0}".format(tweet))
    return tweet


def tokenise(tweet):
    # tokenise and stopword removal
    tweet_words = tokeniser.tokenize(tweet)
    return tweet_words


# Returns a list of common english terms (words)
def initialize_words():
    content = None
    with open('./common_words.txt') as f:  # A file containing common english words
        content = f.readlines()
    return [word.rstrip('\n') for word in content]


wordlist = initialize_words()


def parse_sentence(sentence, wordlist):
    new_sentence = ""  # output
    terms = tokenise(sentence)
    for term in terms:
        if len(term) > 0 and term[0] == '#':  # this is a hashtag, parse it
            new_sentence += parse_tag(term, wordlist)
        else:  # Just append the word
            new_sentence += term
        new_sentence += " "

    return new_sentence.split()


def parse_tag(term, wordlist):
    words = []
    # Remove hashtag, split by dash
    tags = term[1:].split('-')
    for tag in tags:
        word = find_word(tag, wordlist)
        while word != None and len(tag) > 0:
            words.append(word)
            if len(tag) == len(word):  # Special case for when eating rest of word
                break
            tag = tag[len(word):]
            word = find_word(tag, wordlist)
    return " ".join(words)


def find_word(token, wordlist):
    i = len(token) + 1
    while i > 1:
        i -= 1
        if token[:i] in wordlist:
            return token[:i]
    return None


def process_hashtags(tweet):
    return parse_sentence(tweet, wordlist)

def replace_emojis(tweet_words):
    for i in range(len(tweet_words)):
        if tweet_words[i] in emoticons_happy:
            tweet_words[i] = 'happy'
        elif tweet_words[i] in emoticons_sad:
            tweet_words[i] = 'sad'

    return tweet_words

def remove_mentions(tweet_words):
    for i in range(len(tweet_words)):
        if tweet_words[i] == '@':
            del tweet_words[i]

    return tweet_words

def preprocess_tweet(tweet):
    # receives the actual tweet
    tweet = remove_links(tweet)
    # tweet_words = tokenise(tweet)

    # extract words from hashtags
    tweet_words = process_hashtags(tweet)

    # # remove mentions
    # tweet_words = remove_mentions(tweet_words)

    # now remove stopwords
    tweet_words = [word for word in tweet_words if word not in stops]

    # stem
    stems = [stemmer.stem(p) for p in tweet_words]

    # deal with emojis
    tweet_words = replace_emojis(stems)

    # remove non-words
    # but leave in ! and ? as they might be useful for analysis
    tweet_words = re.sub(r"[^\w\s!?]|_", " ", " ".join(tweet_words)).split()

    print("Preprocessed tweet = {0}".format(tweet_words))
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
    print("Reading the training collection")
    tweet_collection_train = read_tweets('./tweetsclassification/Tweets.14cat.train')

    print("Reading the test collection")
    tweet_collection_test = read_tweets('./tweetsclassification/Tweets.14cat.test')

    # careful! bow is built using only the training set
    bow = build_feature_dict(tweet_collection_train)


    build_feature_file('./tweetsclassification/Tweets.14cat.train', './feats.bow', './class_id.txt', 'feats.train.improved')
    build_feature_file('./tweetsclassification/Tweets.14cat.test', './feats.bow', './class_id.txt', 'feats.test.improved')

    #print(evaluate_model('./feats.test', './pred.out', 'Eval.txt'))

    # tweet = '# np Music from Keystone Jaq :) https://t.co/JI3jZSleVM #EDM #house #deephouse #trance #DanceMusic	Music @mariana'
    # print(preprocess_tweet(tweet))



