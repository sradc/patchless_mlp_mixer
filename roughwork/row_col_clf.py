import tensorflow as tf

def RowColClf(height, width, num_classes, Net=MLP, name=None):
    "Rather than average pooling, use MLP for rows, cols, then concat."
    inputs = tf.keras.layers.Input(shape=(height, width))
    rows_out = Net(1)(inputs)
    rows_out = tf.keras.layers.Lambda(lambda x: x[:, :, 0])(rows_out)
    cols_out = tf.keras.layers.Permute([2, 1])(inputs)
    cols_out = Net(1)(cols_out)
    cols_out = tf.keras.layers.Lambda(lambda x: x[:, :, 0])(cols_out)
    concat = tf.keras.layers.Concatenate()([rows_out, cols_out])
    flat = tf.keras.layers.Flatten()(concat)
    y = tf.keras.layers.Dense(num_classes)(flat)
    return tf.keras.Model(inputs=inputs, outputs=y, name=name)

# Put this aside for now, in favour of more general reduction method,
# i.e. -> MLP_Mixer with a small out size.
