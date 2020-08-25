import tensorflow as tf


def my_loss(y_true, y_pred, d):
    SAM = []
    num = y_true.shape[0]
    for i in range(num):
        d = tf.transpose(d)
        A = tf.reduce_sum(tf.multiply(y_pred[i,:], d))
        B = tf.norm(y_pred[i,:], ord = 2)
        C = tf.norm(d, ord = 2)
        defen = tf.div(A, B*C+0.00001)
        # defen = -tf.acos(defen)
        SAM.append(defen)
    print(len(SAM))
    s = tf.nn.top_k(SAM,k=20).values
    sam_loss = tf.reduce_mean(s)
    mse_loss = tf.reduce_mean(tf.square(y_pred - y_true), axis=-1)
    distance_loss = mse_loss + 0.1*sam_loss
    return distance_loss
# MLP as encoder
def MLP_encoder(x, n_hidden, n_output, keep_prob):
    with tf.variable_scope("MLP_encoder"):
        # initializers
        w_init = tf.contrib.layers.xavier_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [x.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(x, w0) + b0
        h0 = tf.nn.leaky_relu(h0,alpha=0.1)
        h0 = tf.nn.dropout(h0, keep_prob)

        # 2nd hidden layer
        w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.leaky_relu(h1,alpha=0.1)
        h1 = tf.nn.dropout(h1, keep_prob)

        # output layer
        wo = tf.get_variable('wo', [h1.get_shape()[1], n_output], initializer=w_init)
        bo = tf.get_variable('bo', [n_output], initializer=b_init)
        output = tf.matmul(h1, wo) + bo
    return output

# MLP as decoder
def MLP_decoder(z, n_hidden, n_output, keep_prob, reuse=False):

    with tf.variable_scope("MLP_decoder", reuse=reuse):
        # initializers
        w_init = tf.contrib.layers.xavier_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [z.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(z, w0) + b0
        h0 = tf.nn.leaky_relu(h0,alpha=0.1)
        h0 = tf.nn.dropout(h0, keep_prob)

        # 2nd hidden layer
        w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.leaky_relu(h1,alpha=0.1)
        h1 = tf.nn.dropout(h1, keep_prob)

        # output layer
        wo = tf.get_variable('wo', [h1.get_shape()[1], n_output], initializer=w_init)
        bo = tf.get_variable('bo', [n_output], initializer=b_init)
        y = tf.tanh(tf.matmul(h1, wo) + bo, name='decode_output')
    return y

# Discriminator
def discriminator(z, n_hidden, n_output, keep_prob, reuse=False):

    with tf.variable_scope("discriminator", reuse=reuse):
        # initializers
        w_init = tf.contrib.layers.xavier_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [z.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(z, w0) + b0
        h0 = tf.nn.leaky_relu(h0,alpha=0.1)
        h0 = tf.nn.dropout(h0, keep_prob)

        # 2nd hidden layer
        w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.leaky_relu(h1,alpha=0.1)
        h1 = tf.nn.dropout(h1, keep_prob)

        # output layer
        wo = tf.get_variable('wo', [h1.get_shape()[1], n_output], initializer=w_init)
        bo = tf.get_variable('bo', [n_output], initializer=b_init)
        y = tf.matmul(h1, wo) + bo
    return tf.sigmoid(y), y

# Gateway
def adversarial_autoencoder(x, z_sample,d, dim_img, dim_z, keep_prob):
    ## Reconstruction Loss
    # encoding
    z = MLP_encoder(x, 400, dim_z, keep_prob)
    # decoding
    y = MLP_decoder(z, 400, dim_img, keep_prob)
    # loss
    R_loss = tf.reduce_mean(tf.reduce_mean(my_loss(x,y,d)))

    ## GAN Loss
    z_real = z_sample#采样得到的
    z_fake = z#重构出来的
    D_real, D_real_logits = discriminator(z_real, 1000, 1, keep_prob)
    D_fake, D_fake_logits = discriminator(z_fake, 1000, 1, keep_prob, reuse=True)

    # discriminator loss
    D_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real_logits)))
    D_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake_logits)))
    D_loss = 0.5*(D_loss_real+D_loss_fake)

    # generator loss
    G_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake_logits)))

    R_loss = tf.reduce_mean(R_loss)
    D_loss = tf.reduce_mean(D_loss)
    G_loss = tf.reduce_mean(G_loss)

    return y, z, R_loss, D_loss, G_loss

