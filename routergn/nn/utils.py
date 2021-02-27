import tensorflow as tf


def norm_values(data, segment_ids, num_segments, name=None):
    values = data[:, 1:]
    if values.shape[1] == 1:
        values = tf.reshape(values, (-1, 1))
    alpha = tf.reshape(data[:, 0], (-1, 1))
    unnormalized_sum = tf.math.unsorted_segment_sum(
        alpha * values, segment_ids, num_segments, name
    )
    norm_ratio = tf.math.unsorted_segment_sum(alpha, segment_ids, num_segments, name)
    return tf.divide(unnormalized_sum, norm_ratio)
