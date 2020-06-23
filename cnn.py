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

import os
import scipy.misc

from ops import *
from utils import *

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class vectorizer(object):
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

    self.batch_size = 256 #バッチサイズ
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

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')

    if not self.y_dim:
      self.g_bn3 = batch_norm(name='g_bn3')

    self.dataset_name = dataset_name #データセットの名前
    self.input_fname_pattern = input_fname_pattern #拡張子のパターン
    self.checkpoint_dir = checkpoint_dir #チェックポイントを吐き出すパス
    self.data_dir = data_dir #データを吐き出すパス

    self.c_dim = 1
    self.grayscale = (self.c_dim == 1) #グレースケールフラグの設定

    self.build_m()

  def build_m(self):
    if self.y_dim: #ラベルの次元が1以上なら
      #プレースホルダーはデータが格納される入れ物。データは未定のままグラフを構築し、具体的な値は実行する時に与える
      self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
    else:
      self.y = None

    #プレースホルダーはデータが格納される入れ物。データは未定のままグラフを構築し、具体的な値は実行する時に与える
    self.inputs = tf.placeholder(tf.float32, [self.batch_size] + [64, 64, 3], name='input_images')

    inputs = self.inputs #プレイスホルダーをローカル変数に代入

    self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z') #zの入れ物を用意

    self.sampler          = self.sampler(self.z, self.y)                #サンプル作成？
    self.V, self.V_logits = self.vectorizer(inputs, self.y, reuse=False)#ベクトライザーの作成？

    self.saver = tf.train.Saver() #全ての変数を保存


  def build_vec(self, config):
    """ベクトライザーをビルドする関数
    """
    #reduce_meanは与えたリストに入っている数値の平均値を求める関数
    print(self.V_logits.shape)
    self.loss = tf.reduce_mean(tf.square(self.z - self.V_logits))
    self.optimizer = tf.train.AdamOptimizer(config.learning_rate)
    self.train = self.optimizer.minimize(self.loss)
    
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()


    #メインのデータをいじるとこ
    for step in range(0, 100):
      sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim)) #一様乱数を生成する

      samples = self.sess.run(self.sampler, feed_dict={self.z: sample_z},)

      sess.run(self.train, feed_dict={ self.inputs: samples, self.z: sample_z })

      print("step: %f , loss: %f"step, sess.run(self.loss, feed_dict={ self.inputs: samples, self.z: sample_z }))


  def verifcation(self):
    """画像のパスを取得し、分割する関数に渡す関数
    """
    with open('./local/eda/z.json') as f:
      data = json.load(f)
    
    paths = ['./local/eda/test_arange_%s.png' % (i) for i in range(step, step + 4)]
    images = [scipy.misc.imread(path).astype(np.float) for path in paths]
    imgs = []
    for image in images:
      imgs.extend(self.img(image))
    imgs = np.array(imgs).astype(np.float32)

    vals = ['./samples/test_arange_%s.png' % (i) for i in range(step, step + 4)]
    z = []
    for val in vals:
      z.extend(data[val])
    z = np.array(z).astype(np.float)


  def sampler(self, z, y=None):
    """生成器
    """
    with tf.variable_scope("sampler") as scope: #generatorという名前空間をオープン
      scope.reuse_variables()

      if not self.y_dim: #y_dimが0なら
        s_h, s_w = self.output_height, self.output_width 
        #conv_out_size_same(x, y)は x / y 以上の最小の整数を返す
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # project `z` and reshape
        h0 = tf.reshape(
            linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'), #linear return matmul(input_, matrix) + bias, (matrix, bias)  
            [-1, s_h16, s_w16, self.gf_dim * 8]) #pooling layer
        h0 = tf.nn.relu(self.g_bn0(h0, train=False)) #activate layer
        h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1') #逆畳み込み
        h1 = tf.nn.relu(self.g_bn1(h1, train=False)) #活性化
        h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
        h2 = tf.nn.relu(self.g_bn2(h2, train=False))
        h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
        h3 = tf.nn.relu(self.g_bn3(h3, train=False))
        h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')
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

  def img(self, img):
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

with tf.Session(config=run_config) as sess:
  dcgan = vectorizer(
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

  dcgan.build_vec(FLAGS)