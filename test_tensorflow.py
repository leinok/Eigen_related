import tensorflow as tf

inputs = tf.constant([1,-2,3,4,-5,6,7,-8,-9,10,11,-12], shape=(4,3), dtype=tf.float32)
weights = tf.constant([.55, -.88, .75, -1.1, -.11, .002], shape=(3, 2), dtype=tf.float32)
bias = tf.constant([3, -2], dtype = tf.float32)
label = tf.constant([1, 0, 1, 1])

relu_outputs = tf.nn.relu(tf.matmul(inputs, weights) + bias)
one_hot = tf.one_hot(label, 2)
predicts = tf.nn.softmax(relu_outputs)
loss = -tf.reduce_mean(one_hot * tf.log(predicts))

d_outputs, d_inputs, d_weights, d_bias = tf.gradients(loss, [relu_outputs, inputs, weights, bias])


with tf.Session() as sess:
    #sess.run(tf.global_variables_initializer())
    loss_np, outputs_np, d_outputs_np, d_inputs_np, d_weights_np, d_bias_np = sess.run([loss, relu_outputs, d_outputs,
                                                                                    d_inputs, d_weights, d_bias])

    print(loss_np)
    print('outputs', outputs_np)
    print('d_outputs', d_outputs_np)
    print('d_inputs', d_inputs_np)
    print('d_weights', d_weights_np)
    print('d_bias', d_bias_np)


        
