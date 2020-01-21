import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from keras.models import *
from keras.layers import *
from keras.utils import *
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import umap
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

class vectorizer(object):

  def build_vec(self):
    """ベクトライザーをビルドする関数
    """
    #プレースホルダーはデータが格納される入れ物。データは未定のままグラフを構築し、具体的な値は実行する時に与える
    self.inputs = tf.placeholder(tf.float32, 100, name='input_images')

    inputs = self.inputs #プレイスホルダーをローカル変数に代入

    self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z') #zの入れ物を用意

    self.V, self.V_logits = self.vectorizer(inputs, self.y, reuse=False)#ディスクリミネーターの作成？

    #reduce_meanは与えたリストに入っている数値の平均値を求める関数
    self.loss = tf.reduce_mean(tf.square(self.z - y_data))
    self.optimizer = tf.train.AdamOptimizerOptimizer(config.learning_rate)
    self.train = self.optimizer.minimize(self.loss)

    self.g_loss_sum = scalar_summary("g_loss", self.g_loss) #g_lossのスカラーの可視化？
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss) #d_lossのスカラーの可視化？

    t_vars = tf.trainable_variables() #trainable_variables()はtrainable=Trueとなっている変数を全て返す.

    self.d_vars = [var for var in t_vars if 'd_' in var.name] #t_varsの中の”d_”で始まるものを選別
    self.g_vars = [var for var in t_vars if 'g_' in var.name] #t_varsの中の”g_”で始まるものを選別

    self.saver = tf.train.Saver() #全ての変数を保存

	def vectorizer(self, image, y=None, reuse=False):
	"""ベクトライザー本体
	"""
	with tf.variable_scope("vectorizer") as scope:
		if reuse:
			scope.reuse_variables()

		if not self.y_dim:
			h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
			h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
			h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
			h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
			h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')

			return tf.nn.sigmoid(h4), h4
		else:
			yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
			x = conv_cond_concat(image, yb)

			h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
			h0 = conv_cond_concat(h0, yb)

			h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
			h1 = tf.reshape(h1, [self.batch_size, -1])      
			h1 = concat([h1, y], 1)
			
			h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
			h2 = concat([h2, y], 1)

			h3 = linear(h2, 1, 'd_h3_lin')
			
			return tf.nn.sigmoid(h3), h3


