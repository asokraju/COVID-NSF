import tensorflow as tf
from tensorflow import keras

import numpy as np
import concurrent.futures 
import time

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if len(gpus) > 0:
#     print(f'GPUs {gpus}')
#     try: tf.config.experimental.set_memory_growth(gpus[0], True)
#     except RuntimeError: pass

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
    return model_clone

def work(model_path, seq):
    # model = clone_model(model)# model_list[model_id]
    # print(model)
    # import tensorflow as tf
    model = tf.keras.models.load_model(model_path)
    return model.predict(seq)

def workers(model, num_of_seq = 4):
    seqences = np.arange(0,num_of_seq*10).reshape(num_of_seq, -1)
    model_savepath = './simple_model.h5'
    model.save(model_savepath)
    path_list = [model_savepath for _ in range(num_of_seq)]

    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:        
        t0 = time.perf_counter()
        # model_list = [clone_model(model) for _ in range(num_of_seq)]
        index_list = np.arange(1, num_of_seq)
        # [clone_model(model) for _ in range(num_of_seq)]
        # print(model_list)
        future_to_samples = {executor.submit(work, path, seq): seq for path, seq in zip(path_list,seqences)}
    Seq_out = []
    for future in concurrent.futures.as_completed(future_to_samples):
        out = future.result()
        Seq_out.append(out)
    t1 = time.perf_counter()
    print(t1-t0)
    return np.reshape(Seq_out, (-1, )), t1-t0



if __name__ == '__main__':
    model = simple_model()
    num_of_seq = 400
    # model_list = [clone_model(model) for _ in range(4)]
    out = workers(model, num_of_seq=num_of_seq)
    print(out)