import  tensorflow as tf
from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from 	tensorflow import keras
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def preprocess(x, y):
    # [0~255] => [-1~1]
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1.
    x = tf.reshape(x, [32 * 32 * 3])
    y = tf.squeeze(y)
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x,y


batchsz = 128
epochsz = 15
# [50k, 32, 32, 3], [10k, 1]
(x, y), (x_val, y_val) = datasets.cifar10.load_data()
print('datasets:', x.shape, y.shape, x_val.shape, y_val.shape, x.min(), x.max())

train_db = tf.data.Dataset.from_tensor_slices((x,y))
train_db = train_db.map(preprocess).shuffle(50000).batch(batchsz).repeat(epochsz)
test_db = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_db = test_db.map(preprocess).batch(batchsz)

sample = next(iter(train_db))
print('batch:', sample[0].shape, sample[1].shape)

class MyDense(layers.Layer):
    # to replace standard layers.Dense()
    def __init__(self, inp_dim, outp_dim):
        super(MyDense, self).__init__()
        self.kernel = self.add_variable('w', [inp_dim, outp_dim])
        self.bias = self.add_variable('b', [outp_dim])

    def call(self, inputs, training=None):
        x = inputs @ self.kernel + self.bias
        return x

class MyNetwork(keras.Model):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.fc1 = MyDense(32*32*3, 256)
        self.fc2 = MyDense(256, 128)
        self.fc3 = MyDense(128, 64)
        self.fc4 = MyDense(64, 32)
        self.fc5 = MyDense(32, 10)

    def call(self, inputs, training=None):
        """
        :param inputs: [b, 32*32*3]
        :param training:
        :return:
        """
        # [b, 32*32*3] => [b, 256]
        x = self.fc1(inputs)
        x = tf.nn.relu(x)
        # [b, 256] => [b, 128]
        x = self.fc2(x)
        x = tf.nn.relu(x)
        # [b, 128] => [b, 64]
        x = self.fc3(x)
        x = tf.nn.relu(x)
        # [b, 64] => [b, 32]
        x = self.fc4(x)
        x = tf.nn.relu(x)
        # [b, 32] => [b, 10]
        x = self.fc5(x)
        return x

network = MyNetwork()
network.build(input_shape=(None, 32*32*3))
network.summary()

optimizer = optimizers.Adam(lr=0.1)

acc_meter = metrics.Accuracy()
loss_meter = metrics.Mean()

for epoch in range(epochsz):
    optimizers.learning_rate = 0.1 * (15 - epoch) / 15

    for step, (x, y) in enumerate(train_db):

        with tf.GradientTape() as tape:
            # [b, 32*32*3] => [b, 10]
            out = network(x)
            loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y, out, from_logits=True))

            loss_regularization = []
            for p in network.trainable_variables:  # 依次对 w 中各项正则化 L2 处理
                loss_regularization.append(tf.nn.l2_loss(p))  # 正则化 L2
            loss_regularization = tf.reduce_sum(tf.stack(loss_regularization))
            loss = loss + 0.001 * loss_regularization

            loss_meter.update_state(loss)


        grads = tape.gradient(loss, network.trainable_variables)
        optimizer.apply_gradients(zip(grads, network.trainable_variables))

        if step % 100== 0:
            print(step, 'loss:', loss_meter.result().numpy())
            loss_meter.reset_states()

        # evaluate
        if step % 500 == 0:
            total, total_correct = 0., 0
            acc_meter.reset_states()

            for step, (x, y) in enumerate(test_db):
                # [b, 32*32*3] => [b, 10]
                out = network(x)
                # [b, 10] => [b]
                pred = tf.argmax(out, axis=1)
                pred = tf.cast(pred, dtype=tf.int64)
                # bool type
                y_cast = tf.argmax(y, axis=1)
                y_cast = tf.cast(y_cast , dtype=tf.int64)
                correct = tf.equal(pred, y_cast)
                # bool tensor => int tensor => numpy
                total_correct += tf.reduce_sum(tf.cast(correct, dtype=tf.int64)).numpy()
                total += x.shape[0]

                acc_meter.update_state(y_cast, pred)

            print(step, 'Evaluate Acc:', total_correct / total, 'acc:', acc_meter.result().numpy())


