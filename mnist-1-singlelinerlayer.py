import input_data
import tensorflow_core as tf

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#启动计算图
sess = tf.InteractiveSession()
#占位符
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])
#权重
W = tf.Variable(tf.zeros([784,10]))
#偏置
b = tf.Variable(tf.zeros([10]))
#初始化
sess.run(tf.initialize_all_variables())
#预测
y = tf.nn.softmax(tf.matmul(x,W) + b)
# 交叉熵作为损失函数
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# 训练---最小化损失函数
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#循环次数
for i in range(1000):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})
#判断是否预测正确
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#计算准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# 打印准确率
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))