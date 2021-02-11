import tensorflow as tf
from tensorflow import keras
import numpy as np


# from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial
import time

def simple_model():
    model = keras.models.Sequential([
        keras.layers.Dense(units = 10, input_shape = [1]),
        keras.layers.Dense(units = 1, activation = 'sigmoid')
    ])
    model.compile(optimizer = 'sgd', loss = 'mean_squared_error')
    return model

def clone_model(model):
    model_clone = tf.keras.models.clone_model(model)
    model_clone.set_weights(model.get_weights())
    model_clone.build((None, 1))
    model_clone.compile(optimizer = 'sgd', loss = 'mean_squared_error')
    return model_clone

def work(model, seq):
    return model.predict(seq)

def work_new(seq):
    model_savepath = './simple_model.h5'
    model = tf.keras.models.load_model(model_savepath)
    return model.predict(seq)

def load_model(model_savepath):
    return tf.keras.models.load_model(model_savepath)

def worker(model, n = 4):
    seqences = np.arange(0,10*n).reshape(n, -1)
    pool = Pool()
    model_savepath = './simple_model.h5'
    model.save(model_savepath)
    # model_list = [load_model(model_savepath) for _ in range(n)]
    # model_list = [clone_model(model) for _ in range(n)]
    # results = pool.map(work, zip(model_list,seqences))
    # path_list = [[model_savepath] for _ in range(n)]
    # print(np.shape(path_list), np.shape(seqences))
    # work_new_partial = partial(work_new, path=model_savepath)
    results = pool.map(work_new,  seqences)
    # partial_work = partial(work, model=model)
    # results = pool.map(partial_work, seqences)
    pool.close()
    pool.join()
    # print(t1-t0)
    return np.reshape(results, (-1, ))



if __name__ == '__main__':

    model = simple_model()
    t0 = time.perf_counter()
    out = worker(model, n=40)
    t1 = time.perf_counter()

    # print(out)
    print(f"time taken {t1 - t0}")