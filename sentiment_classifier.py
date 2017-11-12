#this model will be a simple dnn which extract features from
# bags of words and timestamps to classify comments by success
#no model savings as
import sqlite3
import tensorflow as tf
import random
import nltk
import sqlite3
import traceback
import logging
import time
import random
import datetime
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#TODO: store ngrams in db to allow model storing

nodes_per_layer = 2000
max_results_to_analyze = 10000000
get_newest_results = True #if i cut out some results, this will only get newer results keeping my bot more updated in the meta
stop_word_list = list(nltk.corpus.stopwords.words('english'))
num_of_score_buckets = 10
num_of_features_per_n = 200
num_of_n_for_ngram = 5
subreddit_list = []

n_classes = 2
model_name = "/tmp/sentiment_model.ckpt"

class DNN_sentiment_classifier():
    def __init__(self):
        self.model_needs_retraining = False
        self.border_values = [] # any num above this
        self.n_gram_orders_dict = {}
        #self.input_width = self.get_input_size()
        self.input_width = num_of_features_per_n*(num_of_n_for_ngram - 1)
        self.optimizer, self.cost, self.x, self.y, self.sess, self.prediction, self.saver = self.build_neural_network()

        if self.model_needs_retraining:
            self.train_nn(50)

    def run_text(self, text):
        input_features = self.create_input_features(text)
        return self.sess.run(self.prediction, feed_dict = {self.x:[input_features]})

    def train_nn(self, epochs):
        if len(self.n_gram_orders_dict.keys()) == 0:
            self.read_metadata(num_of_n_for_ngram, num_of_features_per_n)
        self.train_neural_network(epochs, self.optimizer, self.cost, self.x, self.y, self.sess, self.prediction)
        self.save_model()

    def save_model(self):
        save_path = self.saver.save(self.sess, model_name)
        self.save_ngrams()
        print("Model saved in file: %s" % save_path)

    def load_model(self, saver, sess):
        saver.restore(sess, model_name)
        self.load_ngrams()

    def build_neural_network(self):
        start_time = time.time()
        #data = tf.placeholder('float')
        x = tf.placeholder('float', [None, self.input_width])
        y = tf.placeholder('float', [None, n_classes])
        prediction = self.neural_network_model(nodes_per_layer, x)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
        optimizer = tf.train.AdamOptimizer().minimize(cost)
        saver = tf.train.Saver()
        sess = tf.Session()
        try:
            self.load_model(saver, sess)
        except:
            print('initializing')
            traceback.print_exc()
            sess.run(tf.global_variables_initializer())
            self.model_needs_retraining = True
        return optimizer, cost, x, y, sess, prediction,saver

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
        inputs = get_input()
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
            #print(i[0], i[1])
            feature_list.append([self.create_input_features(i), self.create_output_features(i)])

        testing_size = int(test_size*len(inputs))
        train_x = [i[0] for i in feature_list[testing_size:]]
        train_y = [i[1] for i in feature_list[testing_size:]]
        test_x = [i[0] for i in feature_list[:testing_size]]
        test_y = [i[1] for i in feature_list[:testing_size]]
        return train_x, train_y, test_x, test_y

    def create_output_features(self, output):
        if output[1] != 0 and output[1] != 1:
            raise Exception()
        if output[1] == 1:
            return [1, 0]
        else:
            return [0, 1]

    def create_input_features(self, i):
        sentence_features = get_text_features(i[0], self.n_gram_orders_dict)
        return sentence_features

    #build ngrams and rank comments
    #sort the ngrams by how common they are, store the most common ones
    def read_metadata(self, max_n, num_of_features_per_n):
        n_gram_dicts = {}
        score_list = []
        self.n_gram_orders_dict = {}

        for n in range(1, max_n):
            n_gram_dicts.setdefault(n, {})
        res = get_input()
        comments = []
        for r in res:
             comments.append(tuple(remove_stopwords(nltk.tokenize.word_tokenize(r[0].lower()))))
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

    def save_ngrams(self):
        with sqlite3.connect('reddit.db') as conn:
            try:
                conn.execute('drop table sentiment_table_values')
            except:
                pass
            conn.execute('create table if not exists sentiment_table_values (timestamp int, n int, word TEXT, rank int)')
            current_timestamp = int(float(datetime.datetime.now().timestamp()))
            for n in range(1, num_of_n_for_ngram):
                for rank, i in enumerate(self.n_gram_orders_dict[n]):
                    conn.execute('insert into sentiment_table_values values (?, ?, ?, ?)', (current_timestamp, n, i[0], rank))
            conn.commit()

    def load_ngrams(self):
        self.n_gram_orders_dict = {}
        with sqlite3.connect('reddit.db') as conn:
            max_rank = conn.execute('''select max(rank)
            from sentiment_table_values''').fetchone()[0]
            max_n = conn.execute('''select max(n)
            from sentiment_table_values''').fetchone()[0]
            for i in range(1, max_n + 1):
                self.n_gram_orders_dict[i] = ['' for i in range(max_rank+1)]

            for n in range(1, max_n + 1):
                for r in range(max_rank+1):
                    self.n_gram_orders_dict[n][r] = tuple(conn.execute('''select word from sentiment_table_values where rank = ? and n = ?''', (r, n)).fetchone())

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
    word_features = [0 for i in range(len(n_gram_dict.keys())*len(n_gram_dict[1]))]
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

#get parent, child, post data from db
#allows user to only allow data from certai subreddits by passing list of elegible subreddit ids into it
def get_input():
    inputs = []
    df = pd.read_csv('SAD.csv', error_bad_lines=False)
    count = 0
    for index, row in df.iterrows():
        inputs.append([row[3], row[1]])
        count+= 1
        if count > 100:
            break
    return inputs

#testing
if __name__ == '__main__':
    sentiment_classifier = DNN_sentiment_classifier()
    sentiment_classifier.train_nn(10)


