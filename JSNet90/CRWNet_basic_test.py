# CRWNet basic version (test codes)

import tensorflow as tf
import numpy as np
import os
from scipy import ndimage
#import scipy.misc

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

lr = 1e-5
lambd = 0.0001
batch_size = 1
height = 256
width = 256
water_height = 32
water_width = 32
TRAIN_MAX_STEPS = 350000
num_train_sample = 6000
num_class = 2

logs_train_dir = '/home/amax/wyq/shuiyin/sy1205'
train_dir1 = '/home/amax/wyq/shuiyin/isyval.txt'

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

cover = tf.placeholder(tf.float32, [batch_size, height,width, 3])
R, G, B = tf.split(cover, 3, 3)

# 128*128
kernel1 = get_weight('kernel1',[3, 3, 1, 4])
conv1 = tf.nn.conv2d(B, kernel1, strides=[1, 2, 2, 1], padding='SAME')
# conv1 = tf.nn.relu(conv1)

kernel2 = get_weight('kernel2',[3, 3, 4, 8])
conv2 = tf.nn.conv2d(conv1, kernel2, strides=[1, 1, 1, 1], padding='SAME')
# conv2 = tf.nn.relu(conv2)

# 64*64
kernel3 = get_weight('kernel3',[3, 3, 8, 16])
conv3 = tf.nn.conv2d(conv2, kernel3, strides=[1, 2, 2, 1], padding='SAME')
# conv3 = tf.nn.relu(conv3)

kernel4 = get_weight('kernel4',[3, 3, 16, 32])
conv4 = tf.nn.conv2d(conv3, kernel4, strides=[1, 1, 1, 1], padding='SAME')
# conv4 = tf.nn.relu(conv4)

# 32*32
kernel5 = get_weight('kernel5',[3, 3, 32, 64])
conv5 = tf.nn.conv2d(conv4, kernel5, strides=[1, 2, 2, 1], padding='SAME')
# conv5 = tf.nn.relu(conv5)

# concat    [16,32,32,65]
conv5_concat = tf.concat([conv5,watermarking],3)

# 64*64
kernel6 = get_weight('kernel6',[3, 3, 32, 65])
conv6 = tf.nn.conv2d_transpose(conv5_concat, kernel6, output_shape = [1,64,64,32],strides=[1, 2, 2, 1], padding='SAME')
# conv6 = tf.nn.relu(conv6)

kernel7 = get_weight('kernel7',[3, 3, 32, 16])
conv7 = tf.nn.conv2d(conv6, kernel7, strides=[1, 1, 1, 1], padding='SAME')
# conv7 = tf.nn.relu(conv7)

# 128*128
kernel8 = get_weight('kernel8',[3, 3, 8, 16])
conv8 = tf.nn.conv2d_transpose(conv7, kernel8, output_shape = [1,128,128,8],strides=[1, 2, 2, 1], padding='SAME')
# conv8 = tf.nn.relu(conv8)

kernel9 = get_weight('kernel9',[3, 3, 8, 4])
conv9 = tf.nn.conv2d(conv8, kernel9, strides=[1, 1, 1, 1], padding='SAME')
# conv9 = tf.nn.relu(conv9)

# 256*256
kernel10 = get_weight('kernel10',[3, 3, 3, 4])
conv10 = tf.nn.conv2d_transpose(conv9, kernel10, output_shape = [1,256,256,3],strides=[1, 2, 2, 1], padding='SAME')
# conv10 = tf.nn.relu(conv10)

kernelout = get_weight('kernelout',[1, 1, 3, 1])
convout = tf.nn.conv2d(conv10, kernelout, strides=[1, 1, 1, 1], padding='SAME')
Bout = convout
# Bout = tf.nn.relu(convout)

# RG = tf.concat([R, G], 3)
# Oe = tf.concat([RG, Bout], 3)
# R1, G1, B1 = tf.split(Oe, 3, 3)

#decoder  128*128
kernel11 = get_weight('kernel11',[3, 3, 1, 4])
conv11 = tf.nn.conv2d(Bout, kernel11, strides=[1, 2, 2, 1], padding='SAME')
# conv11 = tf.nn.relu(conv11)

kernel12 = get_weight('kernel12',[3, 3, 4, 8])
conv12 = tf.nn.conv2d(conv11, kernel12, strides=[1, 1, 1, 1], padding='SAME')
# conv12 = tf.nn.relu(conv12)

# 64*64
kernel13 = get_weight('kernel13',[3, 3, 8, 16])
conv13 = tf.nn.conv2d(conv12, kernel13, strides=[1, 2, 2, 1], padding='SAME')
# conv13 = tf.nn.relu(conv13)

kernel14 = get_weight('kernel14',[3, 3, 16, 8])
conv14 = tf.nn.conv2d(conv13, kernel14, strides=[1, 1, 1, 1], padding='SAME')
# conv14 = tf.nn.relu(conv14)

# 32*32
kernel15 = get_weight('kernel15',[3, 3, 8, 3])
conv15 = tf.nn.conv2d(conv14, kernel15, strides=[1, 2, 2, 1], padding='SAME')
# conv15 = tf.nn.relu(conv15)

kernel16 = get_weight('kernel16',[3, 3, 3, 3])
conv16 = tf.nn.conv2d(conv15, kernel16, strides=[1, 1, 1, 1], padding='SAME')
# conv16 = tf.nn.relu(conv16)

kernel17 = get_weight('kernel17',[1, 1, 3, 2])
Od= tf.nn.conv2d(conv16, kernel17, strides=[1, 1, 1, 1], padding='SAME')

pred = tf.argmax(Od, dimension=3)

labels = tf.cast(watermarking, tf.int32)
labels = tf.one_hot(labels, depth=num_class)

# loss function
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
# sess.run(tf.global_variables_initializer())#所有节点初始化
ckpt = tf.train.get_checkpoint_state("/home/amax/wyq/shuiyin/sy1205")
saver.restore(sess, ckpt.model_checkpoint_path)

train_count = 0
for step in range(num_train_sample):
    # val_loss = 0
    for j in range(batch_size):
        train_count = train_count % num_train_sample
        train_imc1 = ndimage.imread(train_imList1[train_count])
        train_data_x1[j,:,:,:] = train_imc1
        train_count = train_count + 1
        val_loss,val_loss1,val_loss2 = sess.run([loss,loss1,loss2],feed_dict={cover: train_data_x1})
        # val_loss_mean = val_loss/batch_size

#     tra_lossR = sess.run([R], feed_dict={cover: train_data_x1})
#     tra_lossG = sess.run([G], feed_dict={cover: train_data_x1})
    tra_lossB = sess.run([Bout], feed_dict={cover: train_data_x1})
    # tra_lossR = np.array(tra_lossR)
    # tra_lossG = np.array(tra_lossG)
    tra_lossB = np.array(tra_lossB)
    # tra_lossR = np.reshape(tra_lossR, (256, 256))
    # tra_lossG = np.reshape(tra_lossG, (256, 256))
    tra_lossB = np.reshape(tra_lossB, (256, 256))
    # name1 = str(train_count) + "r" + ".txt"
    # name2 = str(train_count) + "g" + ".txt"
    name3 = str(train_count) + "b" + ".txt"
    # np.savetxt(name1, tra_lossR)
    # np.savetxt(name2, tra_lossG)
    np.savetxt(name3, tra_lossB)

    predout = np.array(sess.run([pred], feed_dict={cover: train_data_x1}))
    get = np.reshape(predout, (32, 32))
    name4 = str(train_count) + "prid" + ".txt"
    np.savetxt(name4, get)

    MMCout = sess.run([watermarking],feed_dict={cover: train_data_x1})
    MMCout = np.array(MMCout)
    goal = np.reshape(MMCout,(32,32))
    name5 = str(train_count) + "goal" + ".txt"
    np.savetxt(name5, goal)
    goal = np.reshape(MMC, (32, 32))

    if (step + 1) % 1 == 0:
        # print('Step %d, validation loss = %.6f' % (step + 1, val_loss))
        print('Step %d, val loss = %.5f,val loss1 = %.5f,val loss2 = %.5f' % (step + 1, val_loss, val_loss1, val_loss2))
sess.close()

