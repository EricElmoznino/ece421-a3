import unittest
from tqdm import tqdm
import os
import tensorflow as tf

import starter_kmeans as km
import starter_gmm as gmm
from helper import load_data
from plotting import *


class Part1(unittest.TestCase):

    def test_2(self):   # includes part 1.1
        self.train = load_data(use_val=False, high_dim=False)
        data_plot_2d(self.train, os.path.join('results', '1_2', 'data.png'))

        ks = [1, 2, 3, 4, 5]
        metrics = [{'Training loss': []} for _ in ks]
        cluster_assignments = []
        epochs = 200
        lr = 0.05

        for k, metric in zip(ks, metrics):
            print('Training with k=%d' % k)
            tf.reset_default_graph()
            x, mu, assignments, loss, optimizer_op = km.build_graph(self.train.shape[-1], k, lr)
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            for _ in tqdm(range(0, epochs + 1)):
                train_loss, _ = sess.run([loss, optimizer_op], feed_dict={x: self.train})
                metric['Training loss'].append(train_loss)
            cluster_assignments.append(sess.run(assignments, feed_dict={x: self.train}))

        for title in metrics[0]:
            line_plot(list(range(0, epochs + 1)), [m[title] for m in metrics],
                      ['k = %d' % k for k in ks], 'epochs', title.split(' ')[1],
                      os.path.join('results', '1_2', title + '.png'))
        for i, k in enumerate(ks):
            k_means_plot_2d(self.train, cluster_assignments[i], k,
                            os.path.join('results', '1_2', 'assignments_k=%d.png' % k))
        for k, assignments, m in zip(ks, cluster_assignments, metrics):
            with open(os.path.join('results', '1_2', 'final_metrics_k=%d.txt' % k), 'w') as f:
                for title in m:
                    f.write('%s: %g\n' % (title, m[title][-1]))
                x_assigned = [self.train[assignments == i] for i in range(k)]
                assigned_percentages = [len(a) / len(self.train) for a in x_assigned]
                assigned_percentages = ['%d:%g' % (i, p) for i, p in enumerate(assigned_percentages)]
                f.write(' '.join(assigned_percentages))

    def test_3(self):
        self.train, self.val = load_data(use_val=True, high_dim=False)

        ks = [1, 2, 3, 4, 5]
        metrics = [{'Training loss': [], 'Validation loss': []} for _ in ks]
        cluster_assignments_train = []
        cluster_assignments_val = []
        epochs = 200
        lr = 0.05

        for k, metric in zip(ks, metrics):
            print('Training with k=%d' % k)
            tf.reset_default_graph()
            x, mu, assignments, loss, optimizer_op = km.build_graph(self.train.shape[-1], k, lr)
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            for _ in tqdm(range(0, epochs + 1)):
                val_loss = sess.run(loss, feed_dict={x: self.val})
                train_loss, _ = sess.run([loss, optimizer_op], feed_dict={x: self.train})
                metric['Training loss'].append(train_loss)
                metric['Validation loss'].append(val_loss)
            cluster_assignments_train.append(sess.run(assignments, feed_dict={x: self.train}))
            cluster_assignments_val.append(sess.run(assignments, feed_dict={x: self.val}))

        for title in metrics[0]:
            line_plot(list(range(0, epochs + 1)), [m[title] for m in metrics],
                      ['k = %d' % k for k in ks], 'epochs', title.split(' ')[1],
                      os.path.join('results', '1_3', title + '.png'))
        for i, k in enumerate(ks):
            k_means_plot_2d(self.val, cluster_assignments_val[i], k,
                            os.path.join('results', '1_3', 'validation_assignments_k=%d.png' % k))
        for k, assignments, m in zip(ks, cluster_assignments_val, metrics):
            with open(os.path.join('results', '1_3', 'final_metrics_k=%d.txt' % k), 'w') as f:
                for title in m:
                    f.write('%s: %g\n' % (title, m[title][-1]))
                x_assigned = [self.val[assignments == i] for i in range(k)]
                assigned_percentages = [len(a) / len(self.val) for a in x_assigned]
                assigned_percentages = ['%d:%g' % (i, p) for i, p in enumerate(assigned_percentages)]
                f.write(' '.join(assigned_percentages))
