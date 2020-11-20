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
import csv

import os
import scipy.misc

from ops import *
from utils import *
from model import *

class VECTORIZER(DCGAN):
  def __init__(self, sess, input_height=108, input_width=108, crop=True,
        batch_size=64, sample_num = 64, output_height=64, output_width=64,
        y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
        gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
        input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None, data_dir='./data'):
    self.build_model()

  def build_model(self):
    """モデルをビルドする関数
    """
    if self.y_dim: #ラベルの次元が1以上なら
      #プレースホルダーはデータが格納される入れ物。データは未定のままグラフを構築し、具体的な値は実行する時に与える
      self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
    else:
      self.y = None

    if self.crop: #cropがTrueならアウトプットサイズを指定
      image_dims = [self.output_height, self.output_width, self.c_dim]
    else:
      image_dims = [self.input_height, self.input_width, self.c_dim]

    #プレースホルダーはデータが格納される入れ物。データは未定のままグラフを構築し、具体的な値は実行する時に与える
    self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images')

    inputs = self.inputs #プレイスホルダーをローカル変数に代入

    self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z') #zの入れ物を用意
    self.z_sum = histogram_summary("z", self.z) #zのヒストグラムの可視化

    self.G                  = self.generator(self.z, self.y)#ジェネレーターの作成
    self.D, self.D_logits   = self.discriminator(inputs, self.y, reuse=False)#ディスクリミネーターの作成？
    self.sampler            = self.sampler(self.z, self.y)#サンプル作成？
    self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)#ディスクリミネーターの作成？
    self.V, self.V_logits   = self.vectorizer(inputs, self.y)#ディスクリミネーターの作成？

    self.d_sum = histogram_summary("d", self.D) #Dのヒストグラムの可視化
    self.d__sum = histogram_summary("d_", self.D_) #D_のヒストグラムの可視化
    self.G_sum = image_summary("G", self.G) #Gのヒストグラムの可視化

    #シグモイド交差エントロピーを返す
    def sigmoid_cross_entropy_with_logits(x, y):
      try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
      except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

    #reduce_meanは与えたリストに入っている数値の平均値を求める関数
    self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
    self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
    self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
    self.loss = tf.reduce_mean(tf.square(self.z - self.V_logits))

    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real) #d_loss_realのスカラーの可視化？
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake) #d_loss_fakeのスカラーの可視化？

    self.d_loss = self.d_loss_real + self.d_loss_fake #ロスの算出

    self.g_loss_sum = scalar_summary("g_loss", self.g_loss) #g_lossのスカラーの可視化？
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss) #d_lossのスカラーの可視化？

    t_vars = tf.trainable_variables() #trainable_variables()はtrainable=Trueとなっている変数を全て返す.

    self.d_vars = [var for var in t_vars if 'd_' in var.name] #t_varsの中の”d_”で始まるものを選別
    self.g_vars = [var for var in t_vars if 'g_' in var.name] #t_varsの中の”g_”で始まるものを選別

    self.saver = tf.train.Saver() #全ての変数を保存


  def train(self, config):
    """モデルのトレーニング
    """
    #reduce_meanは与えたリストに入っている数値の平均値を求める関数
    self.optimizer = tf.train.AdamOptimizer(config.learning_rate)
    self.train = self.optimizer.minimize(self.loss)
    
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    ls = []
    #メインのデータをいじるとこ
    for step in range(0, 1001):
      sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim)) #一様乱数を生成する

      samples = self.sess.run(self.sampler, feed_dict={self.z: sample_z},)

      sess.run(self.train, feed_dict={ self.inputs: samples, self.z: sample_z })
      loss = sess.run(self.loss, feed_dict={ self.inputs: samples, self.z: sample_z })
      print("step: %f , loss: %f" %(step, loss))
      ls.append(float(loss))

      if np.mod(step, 1000) == 0:
          self.save(config.checkpoint_dir, step)
          save_images(samples, image_manifold_size(samples.shape[0]), './local/eda/train_{:02d}.png'.format(step))      

    np.savetxt('./loss_rate_10000.csv', ls)

    real_z = self.verifcation(100)
    samples = self.sess.run(self.sampler, feed_dict={ self.z: real_z},)
    save_images(samples, image_manifold_size(samples.shape[0]), './{}/train_{:02d}.png'.format(config.sample_dir, step))
  

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
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 100, 'd_h4_lin')

        return tf.nn.sigmoid(h4), h4
      else:
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        x = conv_cond_concat(image, yb)
        print("vec-init")
        h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
        h0 = conv_cond_concat(h0, yb)
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
        h1 = tf.reshape(h1, [self.batch_size, -1])      
        h1 = concat([h1, y], 1)
        h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
        h2 = concat([h2, y], 1)
        h3 = linear(h2, 1, 'd_h3_lin')
        
        return tf.nn.sigmoid(h3), h3


#初期パラメーターの定義
flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]") #エポックのサイズ
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]") #adamから持ってきた学習率
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]") #adamの運動量項
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]") #訓練データのサイズ
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]") #バッチのサイズ
flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]") #インプット画像の高さ
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]") #インプット画像の幅
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]") #生成する画像の高さ
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]") #インプット画像の幅
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]") #データセットの名前
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]") #入力画像のパターン
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]") #チェックポイントの保存先のディレクトリ
flags.DEFINE_string("data_dir", "./data", "Root directory of dataset [data]") #データセットのルートディレクトリ
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]") #サンプル画像の保存先
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]") #トレーニングか否か
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]") #トレーニングに対してTrue、テストに対してFalse
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]") #可視化する場合True
flags.DEFINE_integer("generate_test_images", 100, "Number of images to generate during test. [100]") #テスト中に生成する画像の数
FLAGS = flags.FLAGS

#出力先ディレクトリの作成

#入力画像のサイズ補完
pp.pprint(flags.FLAGS.__flags)

#入力画像のサイズ補完
if FLAGS.input_width is None:
  FLAGS.input_width = FLAGS.input_height
if FLAGS.output_width is None:
  FLAGS.output_width = FLAGS.output_height

#出力先ディレクトリの作成
if not os.path.exists(FLAGS.checkpoint_dir):
  os.makedirs(FLAGS.checkpoint_dir)
if not os.path.exists(FLAGS.sample_dir):
  os.makedirs(FLAGS.sample_dir)

#gpuの使用設定
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth=True

#tfセッションの作成、モデルの生成
with tf.Session(config=run_config) as sess:
  vectorize = VECTORIZER(
      sess,
      input_width=FLAGS.input_width,
      input_height=FLAGS.input_height,
      output_width=FLAGS.output_width,
      output_height=FLAGS.output_height,
      batch_size=FLAGS.batch_size,
      sample_num=FLAGS.batch_size,
      z_dim=FLAGS.generate_test_images,
      dataset_name=FLAGS.dataset,
      input_fname_pattern=FLAGS.input_fname_pattern,
      crop=FLAGS.crop,
      checkpoint_dir=FLAGS.checkpoint_dir,
      sample_dir=FLAGS.sample_dir,
      data_dir=FLAGS.data_dir)

  show_all_variables()

vectorize.train(FLAGS)

