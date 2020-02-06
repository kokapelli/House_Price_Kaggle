import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split

from feat_extraction import *

# Advanced regression techniques like random forest and gradient boosting
TRAIN = 1

def dummy_data(data, columns):
    for column in columns:
        data = pd.concat([data, pd.get_dummies(data[column], prefix=column)], axis=1)
        data = data.drop(column, axis=1)
    return data

def get_column_list(data):
    column_list = list()
    for col in data.columns: 
        if(col == 'Id'):
            continue
        column_list.append(col)

    return column_list


if __name__ == "__main__":
    train = pd.read_csv("/Users/Kukus/Desktop/House_Prices_Kaggle/Data/train.csv")
    test = pd.read_csv("/Users/Kukus/Desktop/House_Prices_Kaggle/Data/test.csv")
    train, test = prune_features(train, test)

    id_test = test['Id'].values
    id_test = id_test.reshape(-1, 1)

    y = train['SalePrice'].values
    y = y.astype(float).reshape(-1, 1)
    X = train.drop('SalePrice', axis=1)

    """
    columns = get_column_list(X)
    print(columns)
    X = dummy_data(X, columns)
    test = dummy_data(test, columns)
    print(X.head())
    """
    
    print(X.shape, test.shape)
    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.1, random_state=0)

    seed = 1                        # for reproducible purpose
    input_size = X_train.shape[1]   # number of features
    learning_rate = 0.001           # most common value for Adam
    epochs = 8500

    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(seed)
        np.random.seed(seed)

        X_input = tf.placeholder(dtype=tf.float32, shape=[None, input_size], name='X_input')
        y_input = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y_input')
        
        W1 = tf.Variable(tf.random_normal(shape=[input_size, 1], seed=seed), name='W1')
        b1 = tf.Variable(tf.random_normal(shape=[1], seed=seed), name='b1')
        sigm = tf.nn.sigmoid(tf.add(tf.matmul(X_input, W1), b1), name='pred')
        
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_input,
                                                                    logits=sigm, name='loss'))
        train_steps = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        pred = tf.cast(tf.greater_equal(sigm, 0.5), tf.float32, name='pred') # 1 if >= 0.5
        acc = tf.reduce_mean(tf.cast(tf.equal(pred, y_input), tf.float32), name='acc')
        
        init_var = tf.global_variables_initializer()

    train_feed_dict = {X_input: X_train, y_input: y_train}
    dev_feed_dict = {X_input: X_dev, y_input: y_dev}
    test_feed_dict = {X_input: test} # no y_input since the goal is to predict it

    if(TRAIN):
        sess = tf.Session(graph=graph)
        sess.run(init_var)
        cur_loss = sess.run(loss, feed_dict=train_feed_dict)
        train_acc = sess.run(acc, feed_dict=train_feed_dict)
        test_acc = sess.run(acc, feed_dict=dev_feed_dict)
        print('step 0: loss {0:.5f}, train_acc {1:.2f}%, test_acc {2:.2f}%'.format(
                            cur_loss, 100*train_acc, 100*test_acc))
                            
        for step in range(1, epochs+1):
            sess.run(train_steps, feed_dict=train_feed_dict)
            cur_loss = sess.run(loss, feed_dict=train_feed_dict)
            train_acc = sess.run(acc, feed_dict=train_feed_dict)
            test_acc = sess.run(acc, feed_dict=dev_feed_dict)
            if step%100 != 0: # print result every 100 steps
                continue
        
            print('step {3}: loss {0:.5f}, train_acc {1:.2f}%, test_acc {2:.2f}%'.format(
                       cur_loss, 100*train_acc, 100*test_acc, step))

        y_pred = sess.run(pred, feed_dict=test_feed_dict).astype(int)
        prediction = pd.DataFrame(np.concatenate([id_test, y_pred], axis=1),
                          columns=['Id', 'SalePrice'])
        
        prediction.to_csv("lr-tf-submission.csv",index=False)