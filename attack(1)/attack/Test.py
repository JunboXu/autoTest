from tensorflow.examples.tutorials.mnist import input_data
from generate import *
from fgsmOnNet import *
import time

def getData():
    fashion_minist = input_data.read_data_sets('Fashion-MNIST')
    X_train = fashion_minist.train.images.reshape(-1, 28, 28, 1)
    y_train = fashion_minist.train.labels
    X_test = fashion_minist.test.images.reshape(-1, 28, 28, 1)
    y_test = fashion_minist.test.labels
    return X_train, y_train, X_test, y_test


def formNPY():
    X_train, y_train, X_test, y_test = getData()
    np.save("test_data/test_data.npy",X_test)
    start=time.clock()
    attacks=generate(X_test,[10000,28,28,1])
    end=time.clock()
    print('生成10k对抗样本时间 %s'%(end-start))
    np.save("attack_data/attack_data.npy",attacks)

def readNPY():
    X_test=np.load("test_data/test_data.npy")
    attacks=np.load("attack_data/attack_data.npy")
    print()

def test():
    X_train, y_train, X_test, y_test = getData()
    # Calculate SSIM
    result = generate(X_test[0:100], (100, 28, 28, 1))
    predict(sess,sess_model,X_test[0:100])
    predict(sess,sess_model,result[0:100])
    print()
    # y1=np.argmax(predict(sess,sess_model,X_train[0:100]),axis=1)
    # y2=np.argmax(predict(sess,sess_model,result),axis=1)
    # for i in range(0, 99):
    #     t1 = tf.convert_to_tensor(X_test[i] * 255, dtype=float)
    #     t2 = tf.convert_to_tensor(result[i], dtype=float)
    #     # print(y1[i], y2[i], sess.run(tf.image.ssim(t1, t2, 1)))
    #     print(sess.run(tf.image.ssim(t1, t2, 1)))

# formNPY()
readNPY()
# test()