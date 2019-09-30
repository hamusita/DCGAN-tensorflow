import os
import scipy.misc
import numpy as np

from model import DCGAN
from utils import pp, visualize, to_json, show_all_variables

import tensorflow as tf

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
def main(_):
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
    if FLAGS.dataset == 'mnist':
      dcgan = DCGAN(
          sess,
          input_width=FLAGS.input_width,
          input_height=FLAGS.input_height,
          output_width=FLAGS.output_width,
          output_height=FLAGS.output_height,
          batch_size=FLAGS.batch_size,
          sample_num=FLAGS.batch_size,
          y_dim=10,
          z_dim=FLAGS.generate_test_images,
          dataset_name=FLAGS.dataset,
          input_fname_pattern=FLAGS.input_fname_pattern,
          crop=FLAGS.crop,
          checkpoint_dir=FLAGS.checkpoint_dir,
          sample_dir=FLAGS.sample_dir,
          data_dir=FLAGS.data_dir)
    else:
      dcgan = DCGAN(
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

    if FLAGS.train:
      dcgan.train(FLAGS)
    else:
      if not dcgan.load(FLAGS.checkpoint_dir)[0]:
        raise Exception("[!] Train a model first, then run test mode")
      

    # to_json("./web/js/layers.js", [dcgan.h0_w, dcgan.h0_b, dcgan.g_bn0],
    #                 [dcgan.h1_w, dcgan.h1_b, dcgan.g_bn1],
    #                 [dcgan.h2_w, dcgan.h2_b, dcgan.g_bn2],
    #                 [dcgan.h3_w, dcgan.h3_b, dcgan.g_bn3],
    #                 [dcgan.h4_w, dcgan.h4_b, None])

    # Below is codes for visualization
    OPTION = 1
    visualize(sess, dcgan, FLAGS, OPTION)

if __name__ == '__main__':
  tf.app.run()
