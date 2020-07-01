import  tensorflow as tf
from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from 	tensorflow import keras
# from    keras.models import Sequential
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def preprocess(x, y):
    """
    x is a simple image, not a batch
    """
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [28 * 28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


batchsz = 128
(x, y), (x_val, y_val) = datasets.mnist.load_data()
print('datasets:', x.shape, y.shape, x.min(), x.max())

db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(60000).batch(batchsz)
ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
ds_val = ds_val.map(preprocess).batch(batchsz)

sample = next(iter(db))
print(sample[0].shape, sample[1].shape)

# network = Sequential([ layers.Dense(256, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu),
#                        layers.Dense(128, activation=tf.nn.relu),
#                        layers.Dense(64, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu),
#                        layers.Dense(32, activation=tf.nn.relu),
#                        layers.Dense(10, activation=tf.nn.sigmoid)])

# network = Sequential([layers.Dense(256, activation='relu'),
#                       layers.Dense(128, activation='relu'),
#                       layers.Dropout(0.6),                  #0.6的线会断掉，只保留0.4的线
#                       layers.Dense(64, activation='relu'),
#                       layers.Dropout(0.3),                  #0.3的线会断掉，只保留0.7的线
#                       layers.Dense(32, activation='relu'),
#                       layers.Dense(10)])
# #少量的dropout，会有助于训练学习速度提升，多量的dropout，会增加训练学习难度，但如果训练好了，就会拥有强泛化性能

network = Sequential([layers.Dense(256, activation='relu'),
                      layers.Dense(128, activation='relu'),
                      layers.Dense(64, activation='relu'),
                      layers.Dense(32, activation='relu'),
                      layers.Dense(10)])

network.build(input_shape=(4, 28 * 28))
network.summary()

network.compile(optimizer=optimizers.Adam(lr=0.001),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
                )

network.fit(db, epochs=5,
            validation_data=ds_val,
            validation_freq=2)

network.evaluate(ds_val)

print('predict 结果')
sample = next(iter(ds_val))
x = sample[0]
y = sample[1]  # one-hot
pred = network.predict(x)  # [b, 10]
# convert back to number
y = tf.argmax(y, axis=1)
pred = tf.argmax(pred, axis=1)

print(pred)
print(y)



