# CRWNet basic version (train codes)
import tensorflow as tf
import numpy as np
import os
from scipy import ndimage
#import scipy.misc

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

lr = 1e-6
lambd = 0.0001
batch_size = 4
height = 256
width = 256
water_height = 32
water_width = 32
TRAIN_MAX_STEPS = 1500000
num_train_sample = 6000
num_class = 2
is_train = True

logs_train_dir = '/home/amax/wyq/shuiyin/sy1205'
train_dir1 = '/home/amax/wyq/shuiyin/isytrain.txt'

# randomly generated binary watermark images
MMC = np.random.rand(batch_size, 32, 32, 1)
MMC[MMC > 0.5] = 1
MMC[MMC <= 0.5] = 0
watermarking = tf.cast(MMC, tf.float32)

def get_weight(name,shape):
    var =tf.get_variable(name,shape,initializer=tf.contrib.layers.xavier_initializer())
    # tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambd)(var))
    #  add_to_collection()
    return var

def flist_loader(flist):
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)
    return imlist
train_imList1 = flist_loader(train_dir1)
############################################
# cover image
cover = tf.placeholder(tf.float32, [batch_size, height,width, 3])
# split the cover image into three channels
R, G, B = tf.split(cover, 3, 3)

# embedding the watermark into the B channel
# 128*128
kernel1 = get_weight('kernel1',[3, 3, 1, 4])
conv1 = tf.nn.conv2d(B, kernel1, strides=[1, 2, 2, 1], padding='SAME')
# add BN
# conv1 = tf.layers.batch_normalization(conv1, training=is_train)
# conv1 = tf.nn.relu(conv1)

kernel2 = get_weight('kernel2',[3, 3, 4, 8])
conv2 = tf.nn.conv2d(conv1, kernel2, strides=[1, 1, 1, 1], padding='SAME')
# conv2 = tf.layers.batch_normalization(conv2, training=is_train)
# conv2 = tf.nn.relu(conv2)

kernel3 = get_weight('kernel3',[3, 3, 8, 16])
conv3 = tf.nn.conv2d(conv2, kernel3, strides=[1, 2, 2, 1], padding='SAME')
# conv3 = tf.layers.batch_normalization(conv3, training=is_train)
# conv3 = tf.nn.relu(conv3)

kernel4 = get_weight('kernel4',[3, 3, 16, 32])
conv4 = tf.nn.conv2d(conv3, kernel4, strides=[1, 1, 1, 1], padding='SAME')
# conv4 = tf.layers.batch_normalization(conv4, training=is_train)
# conv4 = tf.nn.relu(conv4)

kernel5 = get_weight('kernel5',[3, 3, 32, 64])
conv5 = tf.nn.conv2d(conv4, kernel5, strides=[1, 2, 2, 1], padding='SAME')
# conv5 = tf.layers.batch_normalization(conv5, training=is_train)
# conv5 = tf.nn.relu(conv5)

#  concat the watermark    [16,32,32,65]
conv5_concat = tf.concat([conv5,watermarking],3)

# 64*64
kernel6 = get_weight('kernel6',[3, 3, 32, 65])
conv6 = tf.nn.conv2d_transpose(conv5_concat, kernel6, output_shape = [4,64,64,32],strides=[1, 2, 2, 1], padding='SAME')
# conv6 = tf.layers.batch_normalization(conv6, training=is_train)
# conv6 = tf.nn.relu(conv6)

kernel7 = get_weight('kernel7',[3, 3, 32, 16])
conv7 = tf.nn.conv2d(conv6, kernel7, strides=[1, 1, 1, 1], padding='SAME')
# conv7 = tf.layers.batch_normalization(conv7, training=is_train)
# conv7 = tf.nn.relu(conv7)

# 128*128
kernel8 = get_weight('kernel8',[3, 3, 8, 16])
conv8 = tf.nn.conv2d_transpose(conv7, kernel8, output_shape = [4,128,128,8],strides=[1, 2, 2, 1], padding='SAME')
# conv8 = tf.layers.batch_normalization(conv8, training=is_train)
# conv8 = tf.nn.relu(conv8)

kernel9 = get_weight('kernel9',[3, 3, 8, 4])
conv9 = tf.nn.conv2d(conv8, kernel9, strides=[1, 1, 1, 1], padding='SAME')
# conv9 = tf.layers.batch_normalization(conv9, training=is_train)
# conv9 = tf.nn.relu(conv9)

# 256*256
kernel10 = get_weight('kernel10',[3, 3, 3, 4])
conv10 = tf.nn.conv2d_transpose(conv9, kernel10, output_shape = [4,256,256,3],strides=[1, 2, 2, 1], padding='SAME')
# conv10 = tf.layers.batch_normalization(conv10, training=is_train)
# conv10 = tf.nn.relu(conv10)

kernelout = get_weight('kernelout',[1, 1, 3, 1])
convout = tf.nn.conv2d(conv10, kernelout, strides=[1, 1, 1, 1], padding='SAME')

# convout = 0.1*0.6*convout
# 16*256*256*3
# Bout = 0.6*(convout-B)+B

Bout = convout
# Bout = tf.nn.relu(convout)

# RG = tf.concat([R, G], 3)
# Oe = tf.concat([RG, Bout], 3)
# R1, G1, B1 = tf.split(Oe, 3, 3)

#decoder  128*128
kernel11 = get_weight('kernel11',[3, 3, 1, 4])
conv11 = tf.nn.conv2d(Bout, kernel11, strides=[1, 2, 2, 1], padding='SAME')
# conv11 = tf.layers.batch_normalization(conv11, training=is_train)
# conv11 = tf.nn.relu(conv11)

#decoder layer 2
kernel12 = get_weight('kernel12',[3, 3, 4, 8])
conv12 = tf.nn.conv2d(conv11, kernel12, strides=[1, 1, 1, 1], padding='SAME')
# conv12 = tf.layers.batch_normalization(conv12, training=is_train)
# conv12 = tf.nn.relu(conv12)

#decoder layer3 64*64
kernel13 = get_weight('kernel13',[3, 3, 8, 16])
conv13 = tf.nn.conv2d(conv12, kernel13, strides=[1, 2, 2, 1], padding='SAME')
# conv13 = tf.layers.batch_normalization(conv13, training=is_train)
# conv13 = tf.nn.relu(conv13)

#decoder layer4
kernel14 = get_weight('kernel14',[3, 3, 16, 8])
conv14 = tf.nn.conv2d(conv13, kernel14, strides=[1, 1, 1, 1], padding='SAME')
# conv14 = tf.layers.batch_normalization(conv14, training=is_train)
# conv14 = tf.nn.relu(conv14)

#decoder layer5 32*32
kernel15 = get_weight('kernel15',[3, 3, 8, 3])
conv15 = tf.nn.conv2d(conv14, kernel15, strides=[1, 2, 2, 1], padding='SAME')
# conv15 = tf.layers.batch_normalization(conv15, training=is_train)
# conv15 = tf.nn.relu(conv15)

#decoder layer5
kernel16 = get_weight('kernel16',[3, 3, 3, 3])
conv16 = tf.nn.conv2d(conv15, kernel16, strides=[1, 1, 1, 1], padding='SAME')
# conv16 = tf.layers.batch_normalization(conv16, training=is_train)
# conv16 = tf.nn.relu(conv16)

#decoder layer7
kernel17 = get_weight('kernel17',[1, 1, 3, 2])
Od= tf.nn.conv2d(conv16, kernel17, strides=[1, 1, 1, 1], padding='SAME')

labels = tf.cast(watermarking, tf.int32)
labels = tf.one_hot(labels, depth=num_class)

# loss funciton
loss1 = tf.reduce_mean(1-tf.image.ssim(B,Bout,max_val = 255))
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = Od, labels = labels)
loss2 = tf.reduce_mean(cross_entropy)
# loss1 = tf.reduce_mean(tf.square(cover-Oe))
# loss2 = tf.reduce_mean(tf.square(watermarking-Od))
loss = 0.7 * loss1+0.3 * loss2
tf.add_to_collection('losses', loss)
loss = tf.add_n(tf.get_collection('losses'))
train_op=tf.train.AdamOptimizer(lr).minimize(loss)

train_data_x1 = np.zeros([batch_size, height, width, 3], dtype=np.float32)

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# ckpt = tf.train.get_checkpoint_state("/home/amax/wyq/shuiyin/wmodelnov24new")
# saver.restore(sess, ckpt.model_checkpoint_path)

train_count = 0
for step in range(TRAIN_MAX_STEPS):
    for j in range(batch_size):
        train_count = train_count % num_train_sample
        train_imc1 = ndimage.imread(train_imList1[train_count])
        train_data_x1[j,:,:,:] = train_imc1
        train_count = train_count + 1

    _, tra_loss,tra_loss1,tra_loss2 = sess.run([train_op, loss,loss1,loss2], feed_dict={cover:train_data_x1})

    if (step + 1) % 10 == 0:
        print('Step %d, train loss = %.5f,loss1 = %.5f,loss2 = %.5f' %(step+1, tra_loss,tra_loss1,tra_loss2))

    if (step + 1) % 5000 == 0:
        checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step+1)

sess.close()
