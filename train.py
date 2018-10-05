import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import preprocessing
import model
import pickle
import random

#load in and randomize training data

print("-------Reading and Processing Data--------\n")   
try:
    with open('data/processed.dat','rb') as f:
        data = pickle.load(f)
except:
    file = 'data/train.csv'
    df = pd.read_csv(file, nrows=20000)
    data = preprocessing.processTrainingBatch(df)
    del df
    pickle.dump(data,open('data/processed.dat','wb'),-1)

zipped = list(zip(data['X'],data['Y']))
random.shuffle(zipped)
inputs, labels = zip(*zipped)

train = {
    'X':np.array(inputs[:int(.8*20000)]),
    'Y':np.array(labels[:int(.8*20000)])
}

test = {
    'X':np.array(inputs[int(.8*20000)+1:]),
    'Y':np.array(labels[int(.8*20000)+1:])
}

    
print("-------Creating Model-------\n")
model = tf.estimator.Estimator(model_fn=model.tf_model_cnn,model_dir="modelData")

#log data
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=1)
tf.logging.set_verbosity(tf.logging.INFO)

#Create train function and train the model
train_fn = tf.estimator.inputs.numpy_input_fn(
    x = train['X'],
    y = train['Y'],
    batch_size=100,
    num_epochs=None,
    shuffle=True
)  

print("-------Training Model-------\n")
train_results = model.train(input_fn=train_fn, steps=int(0.8*20000))
print(train_results)


#load test data and evaluate the model
print("-------Loading Test Data-------\n")

eval_fn = tf.estimator.inputs.numpy_input_fn(
    x=test['X'],
    y=test['Y'],
    num_epochs=1,
    shuffle=False
)

print("-------Evaluating Model-------\n")

eval_results = model.evaluate(input_fn=eval_fn)
print(eval_results)

print("-------Making Predictions-------\n")

predFile = 'data/test.csv'
p = 0.005 #get 1% of the test data set
predict_raw_data = pd.read_csv(predFile, header=0, skiprows=lambda i: i>0 and random.random() > p)
predictData = preprocessing.processTestBatch(predict_raw_data)
pred_fn = tf.estimator.inputs.numpy_input_fn(
    x=predictData['X'],
    y=None,
    num_epochs = 1,
    shuffle=False
)

predict_results = model.predict(input_fn=pred_fn)
results = []
count = 0
for each in predict_results:
    results.append({count:each["classes"]})
    count += 1
print(results)