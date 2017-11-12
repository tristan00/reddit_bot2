#this model will be a simple dnn which extract features from
# bags of words and timestamps to classify comments by success
#no model savings as

import datetime
import logging
import random
import sqlite3
import time

import nltk
import numpy as np
import tensorflow as tf

from reddit_bot2.sentiment_classifier import DNN_sentiment_classifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#TODO: store ngrams in db to allow model storing

nodes_per_layer = 2000
max_results_to_analyze = 100
get_newest_results = True #if i cut out some results, this will only get newer results keeping my bot more updated in the meta
stop_word_list = list(nltk.corpus.stopwords.words('english'))
num_of_score_buckets = 10
num_of_features_per_n = 50
num_of_n_for_ngram = 5
subreddit_list = []

n_classes = 10
model_name = "model_10l_2o.ckpt"

class DNN_comment_classifier():
    def __init__(self):
        self.border_values = [] # any num above this
        self.n_gram_orders_dict = {}
        #self.input_width = self.get_input_size()
        self.input_width = 715
        self.optimizer, self.cost, self.x, self.y, self.sess, self.prediction = self.build_neural_network()
        self.s_classifier = DNN_sentiment_classifier()

    def run_input(self, i):
        return self.sess.run(self.prediction, feed_dict = {self.x:[i]})

    def train_nn(self, epochs):
        self.read_metadata(num_of_n_for_ngram, num_of_features_per_n)
        self.train_neural_network(epochs, self.optimizer, self.cost, self.x, self.y, self.sess, self.prediction)


    def build_neural_network(self):
        start_time = time.time()
        #data = tf.placeholder('float')
        x = tf.placeholder('float', [None, self.input_width])
        y = tf.placeholder('float', [None, n_classes])
        prediction = self.neural_network_model(nodes_per_layer, x)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
        optimizer = tf.train.AdamOptimizer().minimize(cost)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        return optimizer, cost, x, y, sess, prediction

    def neural_network_model(self, nodes_per_layer, x):
        hidden_1_layer = {'weights': tf.Variable(tf.random_normal([self.input_width, nodes_per_layer])),
                          'biases': tf.Variable(tf.random_normal([nodes_per_layer]))}
        hidden_2_layer = {'weights': tf.Variable(tf.random_normal([nodes_per_layer, nodes_per_layer])),
                          'biases': tf.Variable(tf.random_normal([nodes_per_layer]))}
        hidden_3_layer = {'weights': tf.Variable(tf.random_normal([nodes_per_layer, nodes_per_layer])),
                          'biases': tf.Variable(tf.random_normal([nodes_per_layer]))}
        hidden_4_layer = {'weights': tf.Variable(tf.random_normal([nodes_per_layer, nodes_per_layer])),
                          'biases': tf.Variable(tf.random_normal([nodes_per_layer]))}
        hidden_5_layer = {'weights': tf.Variable(tf.random_normal([nodes_per_layer, nodes_per_layer])),
                          'biases': tf.Variable(tf.random_normal([nodes_per_layer]))}
        output_layer = {'weights': tf.Variable(tf.random_normal([nodes_per_layer, n_classes])),
                            'biases': tf.Variable(tf.random_normal([n_classes]))}

        l1 = tf.add(tf.matmul(x, hidden_1_layer['weights']), hidden_1_layer['biases'])
        l1 = tf.nn.relu(l1)
        l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
        l2 = tf.nn.relu(l2)
        l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
        l3 = tf.nn.relu(l3)
        l4 = tf.add(tf.matmul(l3, hidden_4_layer['weights']), hidden_4_layer['biases'])
        l4 = tf.nn.relu(l4)
        l5 = tf.add(tf.matmul(l4, hidden_5_layer['weights']), hidden_5_layer['biases'])
        l5 = tf.nn.relu(l5)
        output = tf.matmul(l5, output_layer['weights']) +  output_layer['biases']
        return output

    def train_neural_network(self, epochs, optimizer, cost, x, y, sess, prediction):
        start_time = time.time()
        batch_size = 10
        hm_epochs = epochs
        inputs = get_db_input()
        random.shuffle(inputs)
        train_x, train_y, test_x, test_y = self.create_feature_sets_and_labels(inputs)

        logger.info('starting training')
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i=0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                _, c = sess.run([optimizer, cost], feed_dict= {x:batch_x, y:batch_y})
                epoch_loss += c
                i += batch_size
            logger.info("Epoch {0} completed out of {1}, loss: {2}".format(epoch, hm_epochs,epoch_loss))

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        accuracy_float = accuracy.eval(session = sess, feed_dict = {x:test_x, y:test_y})
        print('Accuracy:', accuracy_float)
        return sess, prediction, x, y

    def create_feature_sets_and_labels(self, inputs, test_size = .05):
        random.shuffle(inputs)
        feature_list = []

        for i in inputs:
            feature_list.append([self.create_input_features(i), self.create_output_features(i)])

        testing_size = int(test_size*len(inputs))
        train_x = [i[0] for i in feature_list[testing_size:]]
        train_y = [i[1] for i in feature_list[testing_size:]]
        test_x = [i[0] for i in feature_list[:testing_size]]
        test_y = [i[1] for i in feature_list[:testing_size]]
        return train_x, train_y, test_x, test_y

    def create_output_features(self, output):
        output_array = np.zeros(num_of_score_buckets) #[0 for i in range(num_of_score_buckets)]
        for i in range(len(self.border_values)):
            if output[6] < i:
                output_array[i] = 1
                return output_array
        output_array[len(self.border_values)] = 1
        return output_array

    def create_input_features(self, i):
        #TODO: add sentiment data
        sub_list = get_subreddit_list()
        parent_sentiment = self.get_sentiment_classification(i[5])
        child_sentiment = self.get_sentiment_classification(i[14])
        post_sentiment = self.get_sentiment_classification(i[21])
        parent_timestamp_features = create_timestamp_features(i[7])
        child_timestamp_features = create_timestamp_features(i[16])
        post_timestamp_features = create_timestamp_features(i[24])
        subreddit_features = get_subreddit_features(i[2] ,sub_list)
        parent_features = get_text_features(i[5], self.n_gram_orders_dict)
        child_features = get_text_features(i[14], self.n_gram_orders_dict)
        title_features = get_text_features(i[21], self.n_gram_orders_dict)
        input_features = np.concatenate((parent_sentiment, child_sentiment, post_sentiment, parent_timestamp_features,
                                         child_timestamp_features, post_timestamp_features,
                                        subreddit_features, parent_features, child_features,title_features))
        #parent_timestamp_features + child_timestamp_features + post_timestamp_features + subreddit_features + parent_features + child_features + title_features
        return input_features

    #build ngrams and rank comments
    #for each comment, remove stopwords then place into n-grams
    #sort the ngrams by how common they are, store the most common ones
    #TODO: figure out if removing stopwords will significantly change comment meaning
    #TODO: figure out if i should seperate the rankings for title, child and parent
    def read_metadata(self, max_n, num_of_features_per_n):
        n_gram_dicts = {}
        score_list = []
        self.n_gram_orders_dict = {}

        for n in range(1, max_n):
            n_gram_dicts.setdefault(n, {})
        res = get_db_input()
        comments = []
        for r in res:
            comments.append(remove_stopwords(nltk.tokenize.word_tokenize(r[5].lower())))# parent
            comments.append(remove_stopwords(nltk.tokenize.word_tokenize(r[14].lower())))#child
            comments.append(remove_stopwords(nltk.tokenize.word_tokenize(r[21].lower())))#post title
            score_list.append(r[15])
        for comment in comments:
            for n in range(1, max_n):
                if len(comment) >= n:
                    for i in range(len(comment) - n):
                        current_value = n_gram_dicts[n].get(tuple(comment[i:i+n]), 0)
                        n_gram_dicts[n][tuple(comment[i:i+n])] = current_value + 1
                else:
                    break
        for n in range(1, max_n):
            self.n_gram_orders_dict[n] = get_dict_keys_sorted_by_values(n_gram_dicts[n], num_of_features_per_n)
        self.border_values = get_border_values(num_of_score_buckets, score_list)

    def get_sentiment_classification(self, text):
        result = self.s_classifier.run_text([text, None])
        if result[0][0] > result[0][1]:
            return [1, 0]
        else:
            return [0, 1]

#Feature creation methods:
def create_timestamp_features(timestamp):
    datetime_timestamp = datetime.datetime.utcfromtimestamp(float(timestamp))
    hour_feature = [0 for i in range(24)]
    week_day_feature = [0 for i in range(7)]
    hour_feature[datetime_timestamp.hour] = 1
    week_day_feature[datetime_timestamp.weekday()] = 1
    np_array =  np.asarray(hour_feature + week_day_feature)
    return np_array

def get_text_features(text, n_gram_dict):
    word_features = np.zeros(len(n_gram_dict.keys())*len(n_gram_dict[1])) #[0 for i in range(len(n_gram_dict.keys())*len(n_gram_dict[1]))]
    index = 0
    formatted_word = ' '.join(remove_stopwords(nltk.tokenize.word_tokenize(text.lower())))
    for n in n_gram_dict.keys():
        for i in n_gram_dict[n]:
            if ' '.join(i) in formatted_word:
                word_features[index] = 1
            index+= 1
    return word_features


def get_subreddit_features(subreddit, subreddit_list):
    subreddit_features = np.zeros(len(subreddit_list)) #[0 for i in range(len(subreddit_list))]
    subreddit_features[subreddit_list.index(subreddit)] = 1
    return subreddit_features

#Helper methods:
def remove_stopwords(input_list):
    results = []
    for i in input_list:
        if i not in stop_word_list:
            results.append(i)
    return results

def get_dict_keys_sorted_by_values(d, number_to_return, reverse = True):
    sorting_list = []
    for i in d.items():
        sorting_list.append(i)
    sorting_list = sorted(sorting_list, key=lambda x: x[1], reverse = reverse)
    return [i[0] for i in sorting_list][0:number_to_return]

#precalculate border values so that it does not need to store the scores in memory
#returns list of num_of_buckets - 1 len with the values representing border between buckets
def get_border_values(num_of_buckets, score_list):
    border_values = []
    for i in range(1, num_of_buckets):
        border_values.append(np.percentile(score_list, 100*i/num_of_buckets))
    return border_values

#get parent, child, post data from db
#allows user to only allow data from certai subreddits by passing list of elegible subreddit ids into it
def get_db_input():
    with sqlite3.connect('reddit.db') as conn:
        res = conn.execute('''select *
    from comments a join comments b on a.c_id = b.parent_id
    join posts c on a.p_id = c.p_id order by b.submitted_timestamp desc''').fetchall()
        if get_newest_results or max_results_to_analyze > len(res):
            return res[:max_results_to_analyze]
        else:
            random.shuffle(res)
            return res[:max_results_to_analyze]

def get_subreddit_list():
    global subreddit_list
    if len(subreddit_list) == 0:
        with sqlite3.connect('reddit.db') as conn:
            res = conn.execute('select distinct s_id from subreddits').fetchall()
            subreddit_list = [i[0] for i in res]
    return subreddit_list

#testing
if __name__ == '__main__':
    dnn = DNN_comment_classifier()
    print('here')
    dnn.train_nn(50)


