from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class DCGAN(object):
  def __init__(self, sess, input_height=108, input_width=108, crop=True,
        batch_size=64, sample_num = 64, output_height=64, output_width=64,
        y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
        gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
        input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None, data_dir='./data'):
    """
    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    #初期変数の設定
    #-----------------------ここから-------------------------------------------------------------
    self.sess = sess #tfセッション
    self.crop = crop #トリミングフラグ

    self.batch_size = batch_size #バッチサイズ
    self.sample_num = sample_num #サンプルの数？

    self.input_height = input_height #入力の高さ
    self.input_width = input_width #入力の幅
    self.output_height = output_height#入力の高さ
    self.output_width = output_width#出力の幅

    self.y_dim = y_dim #ラベルの数（次元）
    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')

    if not self.y_dim:
      self.d_bn3 = batch_norm(name='d_bn3')

    self.v_bn1 = batch_norm(name='v_bn1')
    self.v_bn2 = batch_norm(name='v_bn2')

    if not self.y_dim:
      self.v_bn3 = batch_norm(name='v_bn3')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')

    if not self.y_dim:
      self.g_bn3 = batch_norm(name='g_bn3')

    self.dataset_name = dataset_name #データセットの名前
    self.input_fname_pattern = input_fname_pattern #拡張子のパターン
    self.checkpoint_dir = checkpoint_dir #チェックポイントを吐き出すパス
    self.data_dir = data_dir #データを吐き出すパス

    #-----------------------ここまで-------------------------------------------------------------

    data_path = os.path.join(self.data_dir, self.dataset_name, self.input_fname_pattern) #データパスを結合
    self.data = glob(data_path) #パスの列挙
    if len(self.data) == 0: #データ数がゼロならエラーを吐く
      raise Exception("[!] No data found in '" + data_path + "'")
    np.random.shuffle(self.data) #データシャッフル
    imreadImg = imread(self.data[0]) #イメージ読み込み
    if len(imreadImg.shape) >= 3: #モノクロかどうかの判別
      self.c_dim = imread(self.data[0]).shape[-1]
    else:
      self.c_dim = 1

    if len(self.data) < self.batch_size: #データよりバッチサイズが大きければエラー
      raise Exception("[!] Entire dataset size is less than the configured batch_size")
    
    self.grayscale = (self.c_dim == 1) #グレースケールフラグの設定

    self.build_model() #モデルのビルド
  #-----------------------ここまでinit-------------------------------------------------------------

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
    self.real_z = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_z')

    inputs = self.inputs #プレイスホルダーをローカル変数に代入

    self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z') #zの入れ物を用意
    self.z_sum = histogram_summary("z", self.z) #zのヒストグラムの可視化

    self.G                  = self.generator(self.z, self.y)#ジェネレーターの作成
    self.D, self.D_logits   = self.discriminator(inputs, self.y, reuse=False)#ディスクリミネーターの作成？
    self.sampler            = self.sampler(self.z, self.y)#サンプル作成？
    self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)#ディスクリミネーターの作成？
    self.V, self.V_logits   = self.vectorizer(inputs, self.y, reuse=False)#ベクトライザーの作成？
    
    self.d_sum = histogram_summary("d", self.D) #Dのヒストグラムの可視化
    self.d__sum = histogram_summary("d_", self.D_) #D_のヒストグラムの可視化
    self.G_sum = image_summary("G", self.G) #Gのヒストグラムの可視化
    self.V_sum = histogram_summary("V", self.V) #Vのヒストグラムの可視化

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
    self.v_loss = tf.reduce_mean(tf.square(self.z - self.V_logits))
    #self.v_loss = tf.reduce_mean(tf.square(self.real_z - self.V_logits))

    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real) #d_loss_realのスカラーの可視化？
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake) #d_loss_fakeのスカラーの可視化？

    self.d_loss = self.d_loss_real + self.d_loss_fake #ロスの算出

    self.g_loss_sum = scalar_summary("g_loss", self.g_loss) #g_lossのスカラーの可視化？
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss) #d_lossのスカラーの可視化？
    self.v_loss_sum = scalar_summary("v_loss", self.v_loss)

    t_vars = tf.trainable_variables() #trainable_variables()はtrainable=Trueとなっている変数を全て返す.

    self.d_vars = [var for var in t_vars if 'd_' in var.name] #t_varsの中の”d_”で始まるものを選別
    self.g_vars = [var for var in t_vars if 'g_' in var.name] #t_varsの中の”g_”で始まるものを選別
    self.v_vars = [var for var in t_vars if 'v_' in var.name]

    self.saver = tf.train.Saver() #全ての変数を保存

  def train(self, config):
    """実際にトレーニングする関数
    """
    #tf.train.AdamOptimizerはAdamアルゴリズムにてminimizeに渡した値を最小化するようトレーニングしてくれる
    d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g_loss, var_list=self.g_vars)

    #全ての変数を初期化する
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    
    self.g_sum = merge_summary([self.z_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])#merge_summaryは与えられたlistのsummaryをmergeする
    self.d_sum = merge_summary([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])#merge_summaryは与えられたlistのsummaryをmergeする

    self.writer = SummaryWriter("./logs", self.sess.graph) #logにsummaryを吐き出す

    sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim)) #一様乱数を生成する
  
    sample_files = self.data[0:self.sample_num] #dataの0からsample_numまでをサンプルとして抽出
    sample = [
        get_image(sample_file,
                  input_height=self.input_height,
                  input_width=self.input_width,
                  resize_height=self.output_height,
                  resize_width=self.output_width,
                  crop=self.crop,
                  grayscale=self.grayscale) for sample_file in sample_files] #dataの0からsample_numまでのイメージを読み込みリスト化
    if (self.grayscale):
      sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
    else:
      sample_inputs = np.array(sample).astype(np.float32)
  
    counter = 1
    start_time = time.time() #時間読み始め

    #今までの学習データの確認
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    #学習の開始
    for epoch in xrange(config.epoch):      
      self.data = glob(os.path.join(config.data_dir, config.dataset, self.input_fname_pattern)) #データの列挙
      np.random.shuffle(self.data) #データシャッフル
      batch_idxs = min(len(self.data), config.train_size) // config.batch_size #データサイズとtrainsizeで小さい方をバッチサイズで切り捨て除算したもの

      #batch_idxs回繰り返す
      for idx in xrange(0, int(batch_idxs)):
        batch_files = self.data[idx * config.batch_size : (idx + 1) * config.batch_size] #バッチサイズごとにイメージを列挙
        batch = [
            get_image(batch_file,
                      input_height=self.input_height,
                      input_width=self.input_width,
                      resize_height=self.output_height,
                      resize_width=self.output_width,
                      crop=self.crop,
                      grayscale=self.grayscale) for batch_file in batch_files] #列挙したものを読み込み
        #npファイルとして画像を読み込み
        if self.grayscale:
          batch_images = np.array(batch).astype(np.float32)[:, :, :, None] 
        else:
          batch_images = np.array(batch).astype(np.float32)

        batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32) #サイズがbatch_size*z_dimの一様乱数を準備(それぞれのイメージに対して潜在変数の用意)

        # Update D network
        _, summary_str = self.sess.run([d_optim, self.d_sum], feed_dict={ self.inputs: batch_images, self.z: batch_z })
        self.writer.add_summary(summary_str, counter)

        # Update G network
        _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={ self.z: batch_z })
        self.writer.add_summary(summary_str, counter)

        # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
        _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={ self.z: batch_z })
        self.writer.add_summary(summary_str, counter)
        
        errD_fake = self.d_loss_fake.eval({ self.z: batch_z })
        errD_real = self.d_loss_real.eval({ self.inputs: batch_images })
        errG = self.g_loss.eval({self.z: batch_z})

        #学習率の表示
        counter += 1
        print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
          % (epoch, config.epoch, idx, batch_idxs,
            time.time() - start_time, errD_fake + errD_real, errG))

        if np.mod(counter, 100) == 1: #counterの100の余剰が1なら
          try:
            samples, d_loss, g_loss = self.sess.run(
              [self.sampler, self.d_loss, self.g_loss],
              feed_dict={
                  self.z: sample_z,
                  self.inputs: sample_inputs,
              },
            )
            save_images(samples, image_manifold_size(samples.shape[0]),
                  './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
            print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) #100回ごとにサンプルを作成
          except:
            print("one pic error!...")

        if np.mod(counter, 500) == 2: #500回ごとにデータをセーブ
          self.save(config.checkpoint_dir, counter)

  def train_vec(self, config):
    #loss = tf.reduce_mean(tf.square(self.z - self.V_logits))
    train = tf.train.AdamOptimizer(config.learning_rate).minimize(self.v_loss)
    
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    self.load(config.checkpoint_dir)[0]

    ls = []

    real_z = self.verifcation(100) #学習データの前処理

    #メインのデータをいじるとこ
    lr = 1000000
    for step in range(0, lr):
      sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim)) #一様乱数を生成する

      samples = self.sess.run(self.sampler, feed_dict={self.z: sample_z},)

      self.sess.run(train, feed_dict={ self.inputs: samples, self.z: sample_z })
      loss = self.sess.run(self.v_loss, feed_dict={ self.inputs: samples, self.z: sample_z })
      print("step: %f , loss: %f" %(step, loss))
      ls.append(float(loss))

      if np.mod(step, 100) == 0:
          self.save(config.checkpoint_dir, step)
          save_images(samples, image_manifold_size(samples.shape[0]), './local/eda/vec_train_{:02d}.png'.format(step))      

    np.savetxt('./loss_rate_%d.csv' % lr, ls)

    for i in range(0, 6400, 64):
      #print("real_z.shape : ", real_z[i:i+64].shape)
      samples = self.sess.run(self.sampler, feed_dict={ self.z: real_z[i:i+64]},)
      save_images(samples, image_manifold_size(samples.shape[0]), './local/eda/spl_generate_%s.png'% int(i/64))
      #print("spl.shape : ", samples.shape)

      vec_z = self.sess.run(self.V_logits, feed_dict={ self.inputs : samples},)
      #print("vec.shape : ", vec_z.shape)
      samples = self.sess.run(self.sampler, feed_dict={ self.z: vec_z},)
      save_images(samples, image_manifold_size(samples.shape[0]), './local/eda/vec_generate_%s.png'% int(i/64))
      print("generate : %s" % int(i/64))

  def verifcation(self, n):
    """画像のパスを取得し、分割する関数に渡す関数
    """
    with open('./samples/z.json') as f:
      data = json.load(f)

    paths = ['./samples/test_arange_%s.png' % (i) for i in range(n)]
    images = [scipy.misc.imread(i).astype(np.float) for i in paths]
    imgs = []
    for image in images:
      imgs.extend(self.cut_img(image))
    imgs = np.array(imgs).astype(np.float32)

    vals = ['./samples/test_arange_%s.png' % (i) for i in range(n)]
    z = []
    for val in vals:
      z.extend(data[val])
    z = np.array(z).astype(np.float)
    print(z.shape, imgs.shape)

    return z

  def cut_img(self, img):
    """画像を64等分して読み込むやつ
    """
    size = 64

    v_size = img.shape[0] // size * size
    h_size = img.shape[1] // size * size
    img = img[:v_size, :h_size]

    v_split = img.shape[0] // size
    h_split = img.shape[1] // size
    out_img = []
    [out_img.extend(np.hsplit(h_img, h_split)) for h_img in np.vsplit(img, v_split)]

    return out_img

  def discriminator(self, image, y=None, reuse=False):
    """ディスクリミネーター本体
    """
    with tf.variable_scope("discriminator") as scope:
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

  def generator(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      if not self.y_dim:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # project `z` and reshape
        self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)
        self.h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(self.h0))
        self.h1, self.h1_w, self.h1_b = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
        h1 = tf.nn.relu(self.g_bn1(self.h1))
        h2, self.h2_w, self.h2_b = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
        h2 = tf.nn.relu(self.g_bn2(h2))
        h3, self.h3_w, self.h3_b = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
        h3 = tf.nn.relu(self.g_bn3(h3))
        h4, self.h4_w, self.h4_b = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)
        return tf.nn.tanh(h4)
      else:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_h4 = int(s_h/2), int(s_h/4)
        s_w2, s_w4 = int(s_w/2), int(s_w/4)

        # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        z = concat([z, y], 1)
        h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
        h0 = concat([h0, y], 1)
        h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin')))
        h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
        h1 = conv_cond_concat(h1, yb)
        h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))
        h2 = conv_cond_concat(h2, yb)
        return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

  #サンプルを吐き出す関数？
  def sampler(self, z, y=None):
    with tf.variable_scope("generator") as scope: #generatorという名前空間をオープン
      scope.reuse_variables()

      if not self.y_dim: #y_dimが0なら
        s_h, s_w = self.output_height, self.output_width # (64, 64)
        #conv_out_size_same(x, y)は x / y 以上の最小の整数を返す
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2) #(32, 32)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2) #(16, 16)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2) #(8, 8)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2) #(4, 4)

        # project `z` and reshape
        h0 = tf.reshape(
            linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'), #linear return matmul(input_, matrix) + bias, (matrix, bias)  (64, 8192)
            [-1, s_h16, s_w16, self.gf_dim * 8]) #pooling layer# (64, 4, 4, 512)
        h0 = tf.nn.relu(self.g_bn0(h0, train=False)) #activate layer
        h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1') # (64, 8, 8, 256)
        h1 = tf.nn.relu(self.g_bn1(h1, train=False)) #活性化
        h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')# (64, 16, 16, 128)
        h2 = tf.nn.relu(self.g_bn2(h2, train=False))
        h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')#(64, 32, 32, 64)
        h3 = tf.nn.relu(self.g_bn3(h3, train=False))
        h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4') #(64, 64, 64, 1)
        return tf.nn.tanh(h4) #処理後のあれを返却
      else:#y_dimが0じゃなければ
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_h4 = int(s_h/2), int(s_h/4)
        s_w2, s_w4 = int(s_w/2), int(s_w/4)

        # yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        z = concat([z, y], 1)
        h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train=False))
        h0 = concat([h0, y], 1)
        h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin'), train=False))
        h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
        h1 = conv_cond_concat(h1, yb)
        h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
        h2 = conv_cond_concat(h2, yb)
        return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

  #ベクトライザー
  def vectorizer(self, image, y=None, reuse=False):
    """ベクトライザー本体
    """
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      # img (64, 64, 64, 1)
      if not self.y_dim:
        h0 = lrelu(conv2d(image, self.df_dim, name='v_h0_conv'))# (64, 32, 32, 64)
        h1 = lrelu(self.v_bn1(conv2d(h0, self.df_dim*2, name='v_h1_conv')))# (64, 16, 16, 128)
        h2 = lrelu(self.v_bn2(conv2d(h1, self.df_dim*4, name='v_h2_conv')))# (64, 8, 8, 256)
        h3 = lrelu(self.v_bn3(conv2d(h2, self.df_dim*8, name='v_h3_conv')))# (64, 4, 4, 512)
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 100, 'v_h4_lin')

        return tf.nn.sigmoid(h4), h4
      else:
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        x = conv_cond_concat(image, yb)
        h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='v_h0_conv'))
        h0 = conv_cond_concat(h0, yb)
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='v_h1_conv')))
        h1 = tf.reshape(h1, [self.batch_size, -1])      
        h1 = concat([h1, y], 1)
        h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'v_h2_lin')))
        h2 = concat([h2, y], 1)

        h3 = linear(h2, 1, 'v_h3_lin')
        
        return tf.nn.sigmoid(h3), h3

  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(self.dataset_name, self.batch_size, self.output_height, self.output_width)
      
  def save(self, checkpoint_dir, step):
    """モデルの保存
    """
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

  def load(self, checkpoint_dir):
    """データの読み込み
    """
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0
