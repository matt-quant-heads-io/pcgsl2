from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
from tensorflow import keras

import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import pickle

"""
Next Steps:
===========
1) Create keras callback for saving best model during training

3) Retrain model w/ one hot obs
4) Run inference.py on 40 randomly generated maps
5) Create numpy archive dataset for stable_baselines .pretrain()
6) Compare inference results from .pretrain() trained model with the pure keras MLP model.

"""

# df = pd.read_csv('narrow_td_onehot_obs.csv')

# file_to_read = open("narrow_td_full.pickle", "rb")

# td_dict = pickle.load(file_to_read)
# df = pd.DataFrame(td_dict)

# df_files = ["narrow_td_onehot_obs_5_goals_lg_part0.csv", "narrow_td_onehot_obs_5_goals_lg_part1.csv", "narrow_td_onehot_obs_5_goals_lg_part2.csv",
#  "narrow_td_onehot_obs_5_goals_lg_part3.csv","narrow_td_onehot_obs_5_goals_lg_part4.csv"]

print(f"here1")
# dfs = []
# for file in df_files:
#     dfs.append(pd.read_csv(file))
#
# print(f"here2")
# df = pd.concat(dfs)#pd.read_csv('narrow_td_onehot_obs_1_goal_lg.csv')
df = pd.read_csv('narrow_td_onehot_obs_50_goals_25_starts.csv')
print(f"df shape: rows: {df.shape[0]} cols: {df.shape[1]}")
print(f"{df.head()}")

# commend this out to not balance data
# df = df[df['target'] > 1].append(df[df['target'] <= 1].iloc[:9000, :])

df = df.sample(frac=1).reset_index(drop=True)
# print(f"{df.head()}")
# print(f"df length {len(df)}")
y_true = df[['target']]
y_true = np_utils.to_categorical(y_true)
df.drop('target', axis=1, inplace=True)
# TODO: uncomment this if you are reading in df from .csv (if using pickle then make sure this is commented b/c column 1 is index
# X = df.iloc[:, 1:]
X = df.iloc[:, :]

train_split = 1.0 #0.9
train_idx = int(len(X) * train_split)
# y_train = y_true.iloc[:train_idx].values.astype('int32')
# y_test = y_true.iloc[train_idx:].values.astype('int32')
y_train = y_true[:train_idx].astype('int32')
y_test = y_true[train_idx:].astype('int32')

X_train = X.iloc[:train_idx, :].values.astype('int32')
X_test = X.iloc[train_idx:, :].values.astype('int32')

input_dim = X_train.shape[1]
nb_classes = y_train.shape[1]

# Here's an MLP (DDMLP)
model = Sequential()
model.add(Dense(4096, input_dim=input_dim))
model.add(Activation('relu'))
# model.add(Dropout(0.15))
model.add(Dense(4096))
model.add(Activation('relu'))
# model.add(Dropout(0.15))
model.add(Dense(8))
model.add(Activation('softmax'))

# model = keras.models.load_model('/Users/matt/gym_pcgrl/gym-pcgil/gym_pcgil/narrow_best_max_acc2.h5')
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(), metrics=[tf.keras.metrics.CategoricalAccuracy()])


# min. loss
# earlyStopping = EarlyStopping(monitor='val_loss', patience=500, verbose=0, mode='min', restore_best_weights=True)
# mcp_save = ModelCheckpoint('narrow_best_min_loss.h5', save_best_only=True, monitor='val_loss', mode='min')

# max. acc
# earlyStopping = EarlyStopping(monitor='accuracy', patience=500, verbose=0, mode='max', restore_best_weights=True)
# mcp_save = ModelCheckpoint('narrow_best_max_acc2.h5', save_best_only=True, monitor='val_categorical_accuracy', mode='max')

# TODO: narrow_best_max_acc_5_goals_500_starts_reg_fit_incomplete.h5 was stopped at 91% acc
mcp_save = ModelCheckpoint('narrow_best_max_acc_50_goals_25_starts_overfit.h5', save_best_only=True, monitor='categorical_accuracy', mode='max')
# mcp_save = ModelCheckpoint('narrow_best_max_acc.h5', save_best_only=True, monitor='val_loss', mode='min')


# model.fit(X_train, y_train, epochs=500, batch_size=1, validation_split=0.25, callbacks=[earlyStopping, mcp_save],
#           verbose=2)
model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split=0.0, callbacks=[mcp_save], verbose=2)

# print("Training...")
# for i in range(3):
#     model.fit(X_train, y_train, epochs=50, batch_size=1, validation_split=0.01, verbose=2)
#
# # print("Generating test predictions...")
# # preds = model.predict_classes(X_test, verbose=0)
#
# df = pd.read_csv('narrow_td_int_obs.csv')
# df = df.sample(frac=1).reset_index(drop=True)
# df = df[df['target'] <= 1]
# y_true = df[['target']]
# y_true = np_utils.to_categorical(y_true)
# df.drop('target', axis=1, inplace=True)
# X = df.iloc[:, 1:]
#
# train_split = 0.9
# train_idx = int(len(X) * train_split)
# # y_train = y_true.iloc[:train_idx].values.astype('int32')
# # y_test = y_true.iloc[train_idx:].values.astype('int32')
# y_train = y_true[:train_idx].astype('int32')
# y_test = y_true[train_idx:].astype('int32')
#
# X_train = X.iloc[:train_idx, :].values.astype('int32')
# X_test = X.iloc[train_idx:, :].values.astype('int32')
#
#
# model.fit(X_train, y_train, epochs=50, batch_size=1, validation_split=0.01, verbose=2)
#
#
# #
# for idx in range(len(X_test)):
#     data = X_test[idx]
#     print(f"data is {data}")
#     # print(f"prediction is {model.predict_classes(np.array([data]), verbose=0)}")
#     print(f"prediction is {np.argmax(model.predict(np.array([data]), verbose=0)[0])}")


# model.save('narrow1_balanced2.h5')

