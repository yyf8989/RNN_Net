import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt

save_ckpt = r"E:\Pycharmprojects\RNN_Net\rnn_ckpt\1_ckpt"


class RNNnet:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
        # NV变型成N（100，28）·V（28）后V的第一层权重：28*128（要变成的权重）
        self.w1 = tf.Variable(tf.truncated_normal([28, 128], dtype=tf.float32, stddev=0.02))
        self.b1 = tf.Variable(tf.zeros([128]))

        self.w2 = tf.Variable(tf.truncated_normal([128, 10], dtype=tf.float32, stddev=0.02))
        self.b2 = tf.Variable(tf.zeros([10]))

    def forward(self):
        # 变形合并前的形状是[100,784]→[100,28,28]，合并后的形状[100*28,28]  //NV→NSV→N(NS)V
        input_x = tf.reshape(self.x, [-1, 28])

        # 第一层计算后的形状[100*28,128]  //N(NS)V
        out_put_1 = tf.nn.relu(tf.matmul(input_x, self.w1) + self.b1)

        # 第一层计算后再变形的形状是从[100*28,128]→[100,28,128]  //N(NS)V→NSV
        # 因为RNN输入的网络结构是NSV结构
        out_put_1 = tf.reshape(out_put_1, [-1, 28, 128])

        # 实例化单层网络记忆细胞的神经元个数
        cell = tf.nn.rnn_cell.BasicLSTMCell(128)

        # def get_cell():
        #     return cell_numbers

        # 叠加多层网络记忆细胞的神经元个数
        # cell_add = tf.nn.rnn_cell.MultiRNNCell([get_cell() for _ in range(2)])#层数

        # 初始化每一批次记忆细胞的状态
        cell_inital_state = cell.zero_state(100, dtype=tf.float32)
        # outputs：LSTM展开后每一个y的值，
        # final_state：最后一个记忆细胞的值 。
        # outputs的值是NSV结构，传入的是记忆细胞的个数、最后输入的y值、初始状态

        outputs, fina_state = tf.nn.dynamic_rnn(cell, out_put_1, initial_state=cell_inital_state,
                                                time_major=False)

        # 矩阵形状转置，NSV转置成SNV，取最后一组（S的最后一步）数据获取结果。就变成了NV
        out_put_2 = tf.transpose(outputs, [1, 0, 2])[-1]

        # 输出最终形状为是从[100,28,128]→[28,100,128]→[1*100,128]=[100,128]  //NSV→N(NS)V→NV
        self.out = tf.matmul(out_put_2, self.w2) + self.b2
        # 上面输出值为softmax前值

        self.output = tf.nn.softmax(self.out)
        # 此处输入softmax分类值

    def backward(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.out, labels=self.y))
        # 定义损失函数，使用softmax交叉熵
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
        # 使用优化器，使用Adam

    def validation(self):
        # 定义验证准确率
        y = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.y, 1))
        # 将输出与标签进行对比，输出bool类型
        self.accuracy = tf.reduce_mean(tf.cast(y, dtype=tf.float32))
        # 计算准确率


if __name__ == '__main__':
    rnn = RNNnet()
    rnn.forward()
    rnn.backward()
    rnn.validation()

    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

    init = tf.global_variables_initializer()
    # 初始化全部变量
    save = tf.train.Saver()
    # 定义save的点

    with tf.Session() as sess:
        sess.run(init)
        # save.restore(sess, save_path=save_ckpt)

        plt.ion()
        for epoch in range(5000):
            # 定义训练次数
            train_x, train_y = mnist.train.next_batch(100)
            _error, _ = sess.run([rnn.loss, rnn.optimizer], feed_dict={rnn.x: train_x, rnn.y: train_y})

            if epoch % 100 == 0:
                test_xs, test_ys = mnist.test.next_batch(100)
                test_output, _accuracy = sess.run([rnn.output, rnn.accuracy],
                                                  feed_dict={rnn.x: test_xs, rnn.y: test_ys})
                test_out = np.argmax(test_output[2])
                # 定义测试输出
                test_ys = np.argmax(test_ys[2])
                # 取出测试的标签
                print('Labels:', test_ys, 'Test:', test_out)
                print("epoch:{0},error:{1:.3f},accuracy:{2:.2f}%".format(epoch, _error, _accuracy * 100))
                save.save(sess, save_path=save_ckpt)
