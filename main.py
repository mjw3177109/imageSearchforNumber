#import tensorflow as tf

#tf.__version__
# from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
# # %matplotlib inline
import numpy as np
# from tensorflow.examples.tutorials import mnist
import math
##tf2.0版本

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data



##导入数据
def load_data():
    mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
    return mnist



##绘制图像方法
def plt_pic(x_train):
    pass



#可视化一个图像
def visulaize_input(img,ax):

    #绘制并输出图像
    ax.imshow(img,cmap="gray")
    print(ax)


    ##对于图像的宽和高，我们输出的数据
    ##便于我们更清晰的知道计算机是如何查看图像的

    width,height =img.shape
    print(width,height)
    #将图像中的具体数值转换成0~1之间的值
    thresh = img.max()/2.5
    #遍历行
    for x in range(width):
        # 遍历列
        for y in range(height):
            #将图像的数值在它对应的位置上标出,且水平垂直居中
            ax.annotate(str(round(img[x][y],2)),xy=(y,x), horizontalalignment='center',verticalalignment='center',color='white' if img[x][y] <thresh else 'black')


    ##假设我们就取出下标为5的样本作为例子




##打印数据集大小
def print_dataset(mnist):
    # 查看训练数据的大小
    print("训练集图像大小{}".format(mnist.train.images.shape))
    print("训练集标签大小{}".format(mnist.train.labels.shape))


    # 查看验证数据的大小
    print("测试集图像大小{}".format(mnist.validation.images.shape))
    print("测试集标签大小{}".format(mnist.validation.labels.shape))


    # 查看测试数据的大小
    print("验证集集图像大小{}".format(mnist.test.images.shape))
    print("验证标签大小{}".format(mnist.test.labels.shape))

    x_train ,y_train =mnist.train.images,mnist.train.labels
    x_valid ,y_valid =mnist.validation.images,mnist.validation.labels
    x_test,y_test=mnist.test.images,mnist.test.labels
    fig=plt.figure(figsize=(10,10))
    ax=fig.add_subplot(111)
    visulaize_input(np.reshape(x_train[5:6],(28,28)),ax)
    # plt.show()


    ###基于多层感知器的tensorflow实现mnist识别
    #多层感知器(Multi-Layer Percetion)模型 中每个节点都是一个感知器 每个感知器的公式是 (x*w)+b 然后再通过激活函数输出两个类别或多个类别
    ##图像大小
    img_size=28*28
    #要预测的类别有几个
    num_classes =10
    ##学习率,也叫作梯度下降值
    learning_rate=0.1
    #迭代次数
    epochs = 100
    #每批次大小
    batch_size=128

    ##创建模型
    #本模型使用softmax多类别分类激活函数 配合交叉熵来计算损失值 然后通过梯度下降来定义优化器,感知器的算法公式一般为(x*w)+b

    #x 表示输入 创建输入占位符 该占位符再训练时 会对每次迭代的数据进行填充
    #None 表示在训练时传入的图像数量 每张大小是img_size

    x=tf.placeholder(tf.float32,[None,img_size])

    #W表示weight 创建权重，初始值都是0 它的大小是(图像的向量大小,图像的总类别个数)
    W=tf.Variable(tf.zeros([img_size,num_classes]))

    #b表示bias 创建偏置项,初始值都是0
    b =tf.Variable(tf.zeros([num_classes]))
    #y 表示计算输出结果,softmax 表示激活函数是多类别分类的输出
    #计算公式 softmax((x*W)+b)
    y = tf.nn.softmax(tf.matmul(x,W)+b)
    #定义输出预测占位符 y_
    y_=tf.placeholder(tf.float32,[None,10])
    ##创建给tensorflow的训练模型时的参数数据字典
    valid_feed_dict ={x:x_valid,y_:y_valid}
    test_feed_dict ={x:x_test,y_:y_test}
    #通过激活函数softmax的交叉熵来定义损失函数
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
    ###定义梯度下降优化器,根据学习率来梯度下降,并且下降过程中,损失值也越来越少
    optimizer =tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    #比较正确的预测结果
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    print(correct_prediction)
    ##计算预测准确率
    accuracy =tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    print(accuracy)

    ##训练模型
    iteration =0

    test_model(accuracy, test_feed_dict)
    #定义训练时的检查点
    # saver =tf.train.Saver() #保存最佳点的模型
    # #创建一个Tensorflow的会话
    # with tf.Session() as sess:
    #     # 初始化全局变量
    #     sess.run(tf.global_variables_initializer())
    #     ##根据每批次训练128个样本 计算出一共需要迭代多少次
    #     batch_count=int(math.ceil(y_train.shape[0]/128.0))
    #     #开始迭代训练样本
    #     for e in range(epochs):
    #
    #         #每个样本都需要在tensorflow的会话中进行运算和训练
    #         for batch_i in range(batch_count):
    #
    #             #样本的索引,间隔是128个
    #             batch_start =batch_i *batch_size
    #             #取出图像样本
    #             batch_x=x_train[batch_start:batch_start + batch_size]
    #
    #             #取出图像对应的标签
    #
    #             batch_y=y_train[batch_start:batch_start+batch_size]
    #
    #             #训练模型
    #             loss,_ =sess.run([cost,optimizer],feed_dict={x:batch_x,y_:batch_y})
    #
    #             ##每训练20次图像后打印一次日志信息
    #             if batch_i %20 ==0:
    #                 print("Epoch: {}/{}".format(e+1,epochs),"Iteration:{}".format(iteration),"Training loss: {:.5f}".format(loss))
    #
    #             iteration+=1
    #
    #             ##每训练128个样本,验证一下训练的模型效果如何,并输出日志
    #             if iteration % batch_size ==0 :
    #                 print(iteration)
    #                 # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    #                 # print(correct_prediction)
    #                 # ##计算预测准确率
    #                 # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #                 valid_acc=sess.run(accuracy,feed_dict={x:x_valid,y_:y_valid})
    #                 #valid_acc =sess.run(accuracy,feed_dict=valid_feed_dict)
    #                 print("Epoch: {}/{}".format(e, epochs), "Iteration:{}".format(iteration),
    #                       "Validation accuracy: {:.5f}".format(valid_acc))
    #
    #     saver.save(sess,"modelstf/mnist_mlp_tf.ckpt")



        #初始化全局变量




#验证模型最后一次最佳点
def test_model(accuracy,test_feed_dict):
    saver=tf.train.Saver()
    with tf.Session() as sess:
        #从训练点恢复
        saver.restore(sess,tf.train.latest_checkpoint('modelstf'))

        #预测测试数据集样本的精确度
        test_acc =sess.run(accuracy,feed_dict=test_feed_dict)
        print("test accuracy:{:.5f}".format(test_acc))





if __name__ == '__main__':
    mnist=load_data()
    print_dataset(mnist)
