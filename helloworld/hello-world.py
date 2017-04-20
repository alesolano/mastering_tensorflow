import tensorflow as tf

##### GRAPH #####
a = tf.placeholder(tf.int32)
b = tf.placeholder(tf.int32)
suma = tf.add(a, b)

##### DATA #####
num1 = 3
num2 = 8

##### SESSION #####
with tf.Session() as sess:
    suma_resultado = sess.run(suma, feed_dict={
        a: num1,
        b: num2
        })

    print("Hello world: la suma de {} y {} es {}".format(num1, num2, suma_resultado))
