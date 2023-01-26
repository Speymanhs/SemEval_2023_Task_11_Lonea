import math
import tensorflow as tf

pi = 3.14159265359


def step_sin_func(x, num_ann, interval):
    return (tf.math.sin(math.pi * (2 * num_ann * x - (2 * interval + 1))/2) + (2 * interval + 1)) / (2*num_ann)


def step_sin_activation(x, num_ann, slope=0.):
    jump = 1. / num_ann
    out = 0.

    # x < 0
    filt = tf.cast((x < 0), tf.float32)
    out = out + filt * x * slope

    # 0 =< x < 1
    for interval in range(num_ann):
        filt = tf.cast((jump * interval <= x), tf.float32) * tf.cast((x < jump * (interval + 1)), tf.float32)
        out = out + filt * step_sin_func(x, num_ann, interval)

    # 1 =< x
    filt = tf.cast((1. <= x), tf.float32)
    out = out + filt + filt * (x - 1) * slope
    return out


def step_discrete_func(inp, num_ann, stretch=1.):
    jump = 1. / num_ann
    out = 0
    for interval in range(num_ann + 1):
        out += (inp >= (interval * stretch + 0.5) * jump) * jump
    return out
