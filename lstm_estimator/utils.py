import tensorflow as tf
import numpy as np

def seq_corr(mode='classification', epsilon=1e-10):
    
    def corr(y1, y2):

        if mode=='classification' or mode=='ordinal_classification':
            y1 = tf.to_float(tf.argmax(y1[0],1))
            y2 = tf.to_float(tf.argmax(y2[0],1))
        if mode=='regression':
            y1, y2 = y1[0], y2[0]

        m1 = y1 - tf.reduce_mean(y1,0)
        m2 = y2 - tf.reduce_mean(y2,0)
        v1 = tf.reduce_sum(m1**2, 0)
        v2 = tf.reduce_sum(m2**2, 0)

        nummerator  = tf.reduce_sum(m1*m2,0)
        denominator = (v1*v2)**0.5

        # N dimensional array holding the correlation for 
        # each output where N is the number of outputs
        n_corr = nummerator/(denominator + epsilon)

        return tf.reduce_mean(n_corr)

    return corr 

def seq_rmse(mode='classification', epsilon=None):
    
    def rmse(y1, y2):
        if mode=='classification' or mode=='ordinal_classification':
            y1 = tf.to_float(tf.argmax(y1[0],1))
            y2 = tf.to_float(tf.argmax(y2[0],1))
        if mode=='regression':
            y1, y2 = y1[0], y2[0]
        return tf.reduce_mean(((y1-y2)**2)**0.5)

    return rmse 

def data_generator(X, Y, max_length=-1):
    def gen():
        while True:
            for x, y in zip(X,Y):
                y_single_sequence = y[None,::]
                x_single_sequence = x[None,::]
                y_single_sequence = y_single_sequence.argmax(-1)

                if max_length==-1:

                    yield x_single_sequence, y_single_sequence

                else:
                    l = y_single_sequence.shape[1]
                    cut = (l//max_length)*max_length

                    x_single_sequence = x_single_sequence[0,:cut]
                    x_batch_sequences = np.stack(np.split(x_single_sequence,cut//max_length, 0))

                    y_single_sequence = y_single_sequence[0,:cut]
                    y_batch_sequences = np.stack(np.split(y_single_sequence,cut//max_length, 0))

                    yield x_batch_sequences, y_batch_sequences

    return gen()

if __name__=='__main__':

    import numpy as np
    np.random.seed(1)
    y_true = np.random.randint(0,6,[1,100,12])
    y_pred = np.random.randint(0,6,[1,100,12])
    y_pred[0,:,0]=y_true[0,:,0]
    y_pred[0,:,4]=y_true[0,:,4]

    y_true_pl = tf.placeholder(tf.float32, y_true.shape)
    y_pred_pl = tf.placeholder(tf.float32, y_true.shape)

    score = seq_corr('regression')
    out = score(y_true_pl, y_pred_pl)

    with tf.Session() as sess:

        feed_dict ={
                y_pred_pl:y_pred,
                y_true_pl:y_true,
                }

        out = sess.run(out, feed_dict=feed_dict)
        print(out)
        print(out.shape)
