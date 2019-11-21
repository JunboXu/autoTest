import numpy as np
import tensorflow as tf
import os

img_width = 28
img_classes = 10
img_chan = 1

def fgm(model, image, eps=0.01, epochs=1):#它的作用是什么？
    xadv = tf.identity(image)

    ybar = model(xadv)
    yshape = ybar.get_shape().as_list()
    ydim = yshape[1]

    indices = tf.argmax(ybar, axis=1)
    target = tf.cond(
        tf.equal(ydim, 1),
        lambda: tf.nn.relu(tf.sign(ybar - 0.5)),
        lambda: tf.one_hot(indices, ydim, on_value=1.0, off_value=0.0))

    if 1 == ydim:
        loss_fn = tf.nn.sigmoid_cross_entropy_with_logits#通过计算网络对于参数的梯度进行噪声的生成即对抗样本的生成。
    else:
        loss_fn = tf.nn.softmax_cross_entropy_with_logits#通过计算网络对于参数的梯度进行噪声的生成即对抗样本的生成。

    noise_fn = tf.sign

    eps = tf.abs(eps)

    def _cond(xadv, i):
        return tf.less(i, epochs)

    def _body(xadv, i):
        ybar, logits = model(xadv, logits=True)
        loss = loss_fn(labels=target, logits=logits)
        dy_dx, = tf.gradients(loss, xadv)
        xadv = tf.stop_gradient(xadv + eps*noise_fn(dy_dx))
        xadv = tf.clip_by_value(xadv, 0., 1.)
        return xadv, i+1

    xadv, _ = tf.while_loop(_cond, _body, (xadv, 0), back_prop=False,
                            name='fast_gradient')
    return xadv

#Construction graph
def model(x, logits=False, training=False):
    with tf.variable_scope('conv0'):
        z = tf.layers.conv2d(x, filters=32, kernel_size=[3, 3],
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('conv1'):
        z = tf.layers.conv2d(z, filters=64, kernel_size=[3, 3],
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('flatten'):
        shape = z.get_shape().as_list()
        z = tf.reshape(z, [-1, np.prod(shape[1:])])

    with tf.variable_scope('mlp'):
        z = tf.layers.dense(z, units=128, activation=tf.nn.relu)
        z = tf.layers.dropout(z, rate=0.25, training=training)

    logits_ = tf.layers.dense(z, units=10, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')

    if logits:
        return y, logits_
    return y


class Model:
    pass


sess_model = Model()

with tf.variable_scope('model'):
    sess_model.x = tf.placeholder(tf.float32, (None, img_width, img_width, img_chan),
                                  name='x')
    sess_model.y = tf.placeholder(tf.float32, (None, img_classes), name='y')
    sess_model.training = tf.placeholder_with_default(False, (), name='mode')

    sess_model.ybar, logits = model(sess_model.x, logits=True, training=sess_model.training)

    with tf.variable_scope('acc'):
        count = tf.equal(tf.argmax(sess_model.y, axis=1), tf.argmax(sess_model.ybar, axis=1))
        sess_model.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')

    with tf.variable_scope('loss'):
        xent = tf.nn.softmax_cross_entropy_with_logits(labels=sess_model.y,
                                                       logits=logits)
        sess_model.loss = tf.reduce_mean(xent, name='loss')

    with tf.variable_scope('train_op'):
        optimizer = tf.train.AdamOptimizer()#此函数是Adam优化算法：是一个寻找全局最优点的优化算法，引入了二次方梯度校正
        sess_model.train_op = optimizer.minimize(sess_model.loss)

    sess_model.saver = tf.train.Saver()#创建saver对象

with tf.variable_scope('model', reuse=True):
    sess_model.fgsm_eps = tf.placeholder(tf.float32, (), name='fgsm_eps')
    sess_model.fgsm_epochs = tf.placeholder(tf.int32, (), name='fgsm_epochs')
    sess_model.x_fgsm = fgm(model, sess_model.x, epochs=sess_model.fgsm_epochs, eps=sess_model.fgsm_eps)

#Retrieve the pre-stored model
sess = tf.InteractiveSession()
saver = tf.train.import_meta_graph('model/mnist.meta') # 加载训练好的模型的meta值
saver.restore(sess, tf.train.latest_checkpoint('./model'))# 加载训练好的参数模型

def make_fgsm(sess, env, X_data, epochs=1, eps=0.01, batch_size=128):#并没有调用生成对抗样本的方法呀？
    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        adv = sess.run(env.x_fgsm, feed_dict={#参数的含义？
            env.x: X_data[start:end],
            env.fgsm_eps: eps,
            env.fgsm_epochs: epochs})
        X_adv[start:end] = adv
    print()

    return X_adv

def evaluate(sess, env, X_data, y_data, batch_size=128):
    """
    Evaluate TF model by running env.loss and env.acc.
    """
    print('\nEvaluating')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    loss, acc = 0, 0

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        cnt = end - start
        batch_loss, batch_acc = sess.run(
            [env.loss, env.acc],
            feed_dict={env.x: X_data[start:end],
                       env.y: y_data[start:end]})
        loss += batch_loss * cnt
        acc += batch_acc * cnt
    loss /= n_sample
    acc /= n_sample

    print(' loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))
    return loss, acc

def train(sess, env, Xdata, ydata, X_valid=None, y_valid=None, epochs=1,
          load=False, shuffle=True, batch_size=128, name='model'):
    """
    Train a TF model by running env.train_op.
    """
    if load:
        if not hasattr(env, 'saver'):
            return print('\nError: cannot find saver op')
        print('\nLoading saved model')
        return env.saver.restore(sess, 'model/{}'.format(name))

    print('\nTrain model')
    n_sample = Xdata.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    for epoch in range(epochs):
        print('\nEpoch {0}/{1}'.format(epoch + 1, epochs))

        if shuffle:
            print('\nShuffling data')
            ind = np.arange(n_sample)
            np.random.shuffle(ind)
            Xdata = Xdata[ind]
            ydata = ydata[ind]

        for batch in range(n_batch):
            print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
            begin = batch * batch_size
            ending = min(n_sample, begin + batch_size)
            sess.run(env.train_op, feed_dict={env.x: Xdata[begin:ending],
                                              env.y: ydata[begin:ending],
                                              env.training: True})
        if X_valid is not None:
            evaluate(sess, env, X_valid, y_valid)

    if hasattr(env, 'saver'):#在此处保存模型吧？
        print('\n Saving model')
        os.makedirs('model', exist_ok=True)
        env.saver.save(sess, 'model/{}'.format(name))

def predict(sess, env, X_data, batch_size=128):#判定图片的依据是什么？
    """
    Do inference by running env.ybar.
    """
    print('\nPredicting')
    n_classes = env.ybar.get_shape().as_list()[1]

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    yval = np.empty((n_sample, n_classes))

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        y_batch = sess.run(env.ybar, feed_dict={env.x: X_data[start:end]})
        yval[start:end] = y_batch
    print()
    return yval
