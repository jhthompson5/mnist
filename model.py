import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle
import preprocessing


    #Create structure for convolutional layer
def tf_model_cnn(features,labels,mode):
    input_layer = tf.reshape(features,[-1,28,28,1])
    input_layer = tf.cast(input_layer,tf.float32)
    conv1 = tf.layers.conv2d(
        inputs = input_layer,
        filters = 32,
        kernel_size = [5,5],
        padding="same",
        activation=tf.nn.relu
    )
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2,2],
        strides=2
    )
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu
    )
    pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides=2)

    #reshape pool2 layer to flat with size [batchsize,7*7*64]
    pool2_flat = tf.reshape(pool2,[-1,7*7*64])

    #Create dense layer to classify features
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    
    #final logit output layer
    logits = tf.layers.dense(inputs=dense, units=10)
    predictions = {
        "classes" : tf.argmax(input=logits, axis=1),
        "probabilities" : tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops)
    
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

