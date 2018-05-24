from __future__ import absolute_import
import mxnet as mx
import mxnet.symbol as mxs
import numpy as np
import logging
from pair_iterator import PairDataIter

logging.basicConfig(level=logging.DEBUG)


def contrastive_metric(label, pred):
    mask_positive = pred < 1
    mask_negative = pred >= 1
    pred[mask_negative] = 0
    pred[mask_positive] = 1
    return np.mean(label == pred)


def siamese_simp_net():
    def conv_bn_relu_pool_siamese(input_a, input_b, kernel, num_filter, pad, stride, name_postfix, use_pooling=False,
                                  p_kernel=None, p_stride=None, use_batch_norm=True):
        conv_weight = mxs.Variable(name='conv' + name_postfix + '_weight')
        conv_bias = mxs.Variable(name='conv' + name_postfix + '_bias')
        conv_a = mxs.Convolution(data=input_a, kernel=kernel, num_filter=num_filter, pad=pad, stride=stride,
                                 name='conv' + name_postfix + "_a", weight=conv_weight, bias=conv_bias)
        conv_b = mxs.Convolution(data=input_b, kernel=kernel, num_filter=num_filter, pad=pad, stride=stride,
                                 name='conv' + name_postfix + "_b", weight=conv_weight, bias=conv_bias)
        if use_batch_norm:
            bn_gamma = mxs.Variable(name='bn' + name_postfix + '_gamma')
            bn_beta = mxs.Variable(name='bn' + name_postfix + '_beta')
            bn_moving_mean = mxs.Variable(name='bn' + name_postfix + '_moving_mean')
            bn_moving_var = mxs.Variable(name='bn' + name_postfix + '_moving_var')
            batch_norm_a = mxs.BatchNorm(data=conv_a, name='bn' + name_postfix + '_a', gamma=bn_gamma, beta=bn_beta,
                                         moving_mean=bn_moving_mean, moving_var=bn_moving_var)
            batch_norm_b = mxs.BatchNorm(data=conv_b, name='bn' + name_postfix + '_b', gamma=bn_gamma, beta=bn_beta,
                                         moving_mean=bn_moving_mean, moving_var=bn_moving_var)
        else:
            batch_norm_a = conv_a
            batch_norm_b = conv_b
        relu_a = mxs.relu(data=batch_norm_a, name='relu' + name_postfix + '_a')
        relu_b = mxs.relu(data=batch_norm_b, name='relu' + name_postfix + '_b')
        if use_pooling:
            out_a = mxs.Pooling(data=relu_a, kernel=p_kernel, pool_type='max', stride=p_stride,
                                name='pool' + name_postfix + '_a')
            out_b = mxs.Pooling(data=relu_b, kernel=p_kernel, pool_type='max', stride=p_stride,
                                name='pool' + name_postfix + '_b')
        else:
            out_a = relu_a
            out_b = relu_b
        return out_a, out_b

    data_a = mxs.Variable('data_a')
    data_b = mxs.Variable('data_b')
    c1_a, c1_b = conv_bn_relu_pool_siamese(data_a, data_b, kernel=(3, 3), num_filter=64, pad=(1, 1), stride=(1, 1),
                                           name_postfix='1', use_pooling=False)
    c1_0_a, c1_0_b = conv_bn_relu_pool_siamese(c1_a, c1_b, kernel=(3, 3), num_filter=32, pad=(1, 1), stride=(1, 1),
                                               name_postfix='1_0', use_pooling=False)
    c2_a, c2_b = conv_bn_relu_pool_siamese(c1_0_a, c1_0_b, kernel=(3, 3), num_filter=32, pad=(1, 1), stride=(1, 1),
                                           name_postfix='2', use_pooling=False)
    c2_1_a, c2_1_b = conv_bn_relu_pool_siamese(c2_a, c2_b, kernel=(3, 3), num_filter=32, pad=(1, 1), stride=(1, 1),
                                               name_postfix='2_1', use_pooling=True, p_kernel=(2, 2), p_stride=(2, 2))
    c2_2_a, c2_2_b = conv_bn_relu_pool_siamese(c2_1_a, c2_1_b, kernel=(3, 3), num_filter=32, pad=(1, 1), stride=(1, 1),
                                               name_postfix='2_2', use_pooling=False)
    c3_a, c3_b = conv_bn_relu_pool_siamese(c2_2_a, c2_2_b, kernel=(3, 3), num_filter=32, pad=(1, 1), stride=(1, 1),
                                           name_postfix='3', use_pooling=False)
    # conv4
    conv4_weight = mxs.Variable(name='conv4_weight')
    conv4_bias = mxs.Variable(name='conv4_bias')
    conv4_a = mxs.Convolution(data=c3_a, kernel=(3, 3), num_filter=64, pad=(1, 1), stride=(1, 1),
                              name='conv4_a', weight=conv4_weight, bias=conv4_bias)  # xavier
    conv4_b = mxs.Convolution(data=c3_b, kernel=(3, 3), num_filter=64, pad=(1, 1), stride=(1, 1),
                              name='conv4_b', weight=conv4_weight, bias=conv4_bias)  # xavier
    maxp4_a = mxs.Pooling(data=conv4_a, kernel=(2, 2), pool_type='max', stride=(2, 2), name='pool4_a')
    maxp4_b = mxs.Pooling(data=conv4_b, kernel=(2, 2), pool_type='max', stride=(2, 2), name='pool4_b')
    bn4_gamma = mxs.Variable(name='bn4_gamma')
    bn4_beta = mxs.Variable(name='bn4_beta')
    bn4_moving_mean = mxs.Variable(name='bn4_moving_mean')
    bn4_moving_var = mxs.Variable(name='bn4_moving_var')
    batch_norm_4_a = mxs.BatchNorm(data=maxp4_a, name='bn4_a', gamma=bn4_gamma, beta=bn4_beta,
                                   moving_mean=bn4_moving_mean, moving_var=bn4_moving_var)
    batch_norm_4_b = mxs.BatchNorm(data=maxp4_b, name='bn4_b', gamma=bn4_gamma, beta=bn4_beta,
                                   moving_mean=bn4_moving_mean, moving_var=bn4_moving_var)
    relu4_a = mxs.relu(data=batch_norm_4_a, name='relu4')
    relu4_b = mxs.relu(data=batch_norm_4_b, name='relu4')
    c4_1_a, c4_1_b = conv_bn_relu_pool_siamese(relu4_a, relu4_b, kernel=(3, 3), num_filter=64, pad=(1, 1),
                                               stride=(1, 1),
                                               name_postfix='4_1', use_pooling=False)
    c4_2_a, c4_2_b = conv_bn_relu_pool_siamese(c4_1_a, c4_1_b, kernel=(3, 3), num_filter=64, pad=(1, 1), stride=(1, 1),
                                               name_postfix='4_2', use_pooling=True, p_kernel=(2, 2), p_stride=(2, 2))
    c4_0_a, c4_0_b = conv_bn_relu_pool_siamese(c4_2_a, c4_2_b, kernel=(3, 3), num_filter=128, pad=(1, 1), stride=(1, 1),
                                               name_postfix='4_0', use_pooling=False)
    cccp4_a, cccp4_b = conv_bn_relu_pool_siamese(c4_0_a, c4_0_b, kernel=(1, 1), num_filter=256, pad=[], stride=(1, 1),
                                                 name_postfix='_cccp4', use_pooling=False, use_batch_norm=False)
    cccp5_a, cccp5_b = conv_bn_relu_pool_siamese(cccp4_a, cccp4_b, kernel=(1, 1), num_filter=64, pad=[], stride=(1, 1),
                                                 name_postfix='_cccp5', use_pooling=True, p_kernel=(2, 2),
                                                 p_stride=(2, 2), use_batch_norm=False)
    cccp6_a, cccp6_b = conv_bn_relu_pool_siamese(cccp5_a, cccp5_b, kernel=(3, 3), num_filter=64, pad=(2, 2),
                                                 stride=(1, 1), name_postfix='_cccp6', use_pooling=False,
                                                 use_batch_norm=False)
    flat_a = mxs.flatten(cccp6_a)
    flat_b = mxs.flatten(cccp6_b)
    return flat_a, flat_b


def siamese():
    labels = mxs.Variable(name='label')
    flat_a, flat_b = siamese_simp_net()
    distance = mxs.sqrt(mxs.sum(mxs.square(flat_a - flat_b), axis=1))
    cl1 = labels * mxs.square(distance)
    cl2 = (1 - labels) * mxs.square(mxs.maximum(1 - distance, 0))
    contrastive_loss = mxs.MakeLoss(mxs.mean(cl1 + cl2))
    distance_output = mxs.BlockGrad(distance, name='distance')
    flat_a_output = mxs.BlockGrad(flat_a)
    flat_b_output = mxs.BlockGrad(flat_b)
    sym = mx.sym.Group([contrastive_loss, distance_output, flat_a_output, flat_b_output])
    mod = mx.mod.Module(symbol=sym, context=mx.gpu(), data_names=['data_a', 'data_b'], label_names=['label'])
    return mod


mod = siamese()
print("Bind")

mod.bind(data_shapes=[mx.io.DataDesc('data_a', (128, 3, 32, 32)), mx.io.DataDesc('data_b', (128, 3, 32, 32))],
         label_shapes=[mx.io.DataDesc('label', (128,))])
print("Initialize")
init = mx.initializer.Mixed(["(conv1|conv1_0|conv3|conv4|conv4_1|conv4_2|conv4_0|cccp4|cccp5|cccp6)_weight",
                             "(cccp4|cccp5|cccp6)_bias", "conv2|conv2_1|conv2_2", ".*"],
                            [mx.initializer.Xavier(), mx.initializer.Zero(), mx.initializer.Normal(),
                             mx.initializer.Uniform()])
mod.init_params(init)

train_iter = PairDataIter(batch_size=1024)

eval_metrics = [
    mx.metric.CustomMetric(contrastive_metric, name='contrastive_accuracy', output_names=['distance_output'],
                           label_names=['label'])]
batch_end_callbacks = [mx.callback.Speedometer(128, 1)]
print("Start learning")
try:
    mod.fit(train_iter,
            optimizer='adam',
            optimizer_params={'learning_rate': 0.0001},
            batch_end_callback=batch_end_callbacks,
            eval_metric=eval_metrics,
            num_epoch=1,
            initializer=init
            )
except KeyboardInterrupt:
    mod.save_checkpoint('/home/user/lobanov/mxnet_learning/data/checkpoints/siamese/siamese', 0)
pass

mx.symbol.Convolution()
