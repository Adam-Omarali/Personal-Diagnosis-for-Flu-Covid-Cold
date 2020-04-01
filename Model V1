import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, Normalizer

csv_file = tf.keras.utils.get_file('Train_Medicine', "https://docs.google.com/spreadsheets/d/e/2PACX-1vSyAaBwmYMGI3CLNfOgczOKPHz9PY5_DmyxbPJtTXxTvDJt5ZCrgdsQbNUvClkRqgH-E0NFRgVTv_Dw/pub?gid=0&single=true&output=csv"
)
df = pd.read_csv(csv_file)
df = df.drop(columns=(["Cold", "Flu", "COVID-19"]))

array = df.values
X = array[:, 0:57]
Y = array[:, 57:]

scaler = Normalizer().fit(X)
normalizedX = scaler.transform(X)

def get_compiled_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='relu')
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model


model = get_compiled_model()
model.fit(X, Y, batch_size=1, epochs=15)
