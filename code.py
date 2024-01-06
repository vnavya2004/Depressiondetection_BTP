import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from keras import layers
from keras import losses
from keras.models import load_model
from tensorboard import notebook
import matplotlib.pyplot as plt
import datetime

print("Version: ", tf.__version__)
train_ds, val_ds, test_ds = tfds.load(
    name="imdb_reviews",
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)
text_vectorizer = layers.TextVectorization(
     output_mode='multi_hot',
     max_tokens=2500
     )

features = train_ds.map(lambda x, y: x)

text_vectorizer.adapt(features)
def preprocess(x):
  return text_vectorizer(x)

inputs = keras.Input(shape=(1,), dtype='string')

outputs = layers.Dense(1)(preprocess(inputs))

model = keras.Model(inputs, outputs)
model.compile(
    optimizer='adam',
    loss=losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
    )
model.summary()
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
epochs = 10

history = model.fit(
    train_ds.shuffle(buffer_size=10000).batch(512),
    epochs=epochs,
    validation_data=val_ds.batch(512),
    verbose=1)
len(history.history['loss'])
results = model.evaluate(test_ds.batch(512), verbose=2)

for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))
history_dict = history.history
history_dict.keys()
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

examples = [
   "The movie was great!",
  "The movie was okay.",
  "The movie was terrible..."
]

probability_model = keras.Sequential([
                        model,
                        layers.Activation('sigmoid')
                        ])
probability_model.predict(examples)
