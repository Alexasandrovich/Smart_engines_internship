import mxnet as mx
import mxnet.symbol as mxs
import cv2
import numpy as np


PIXEL_MEANS = [103.06, 115.90, 123.15]


def infer_shape(sym, print_arguments=False, **kwargs):
    if 'data' not in kwargs:
        kwargs['data'] = (1, 3, 1080, 1920)
    if kwargs['data'] is None:
        del kwargs['data']
    args = sym.list_arguments()
    aux = sym.list_auxiliary_states()
    out = sym.list_outputs()
    args_shapes, out_shapes, aux_shapes = sym.infer_shape(**kwargs)
    if print_arguments:
        print('Arguments: ')
        for name, shape in zip(args, args_shapes):
            print('{}: {}'.format(name, shape))
        print('Auxiliary: ')
        for name, shape in zip(aux, aux_shapes):
            print('{}: {}'.format(name, shape))
    print('Outputs: ')
    for name, shape in zip(out, out_shapes):
        print('{}: {}'.format(name, shape))
    print('')


def transform(im, pixel_means=PIXEL_MEANS):
    """
    transform into mxnet tensor
    substract pixel size and transform to correct format
    :param im: [height, width, channel] in BGR
    :param pixel_means: [B, G, R pixel means]
    :return: [batch, channel, height, width]
    """
    im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))
    for i in range(3):
        im_tensor[0, i, :, :] = im[:, :, 2 - i] - pixel_means[2 - i]
    return im_tensor


def embedder(data, suffix=''):
    conv1 = mx.symbol.Convolution(name='conv1' + suffix, data=data, num_filter=64, pad=(3, 3), kernel=(7, 7),
                                  stride=(2, 2),
                                  no_bias=True)

    bn_conv1 = mx.symbol.BatchNorm(name='bn_conv1' + suffix, data=conv1)
    conv1_relu = mx.symbol.Activation(name='conv1_relu' + suffix, data=bn_conv1, act_type='relu')
    pool1 = mx.symbol.Pooling(name='pool1' + suffix, data=conv1_relu, pooling_convention='full', pad=(0, 0),
                              kernel=(3, 3), stride=(2, 2), pool_type='max')
    conv2 = mx.symbol.Convolution(name='conv2' + suffix, data=pool1, num_filter=16, pad=(0, 0),
                                          kernel=(1, 1), stride=(1, 1), no_bias=True)
    conv2_relu = mx.symbol.Activation(name='res2a_branch2a_relu' + suffix, data=conv2, act_type='relu')
    emb = mx.symbol.flatten(conv2_relu)
    return emb


im_1 = cv2.imread('/media/user/COMMON/lobanov/data/a7.#.rec.notrash/3.4/00002118.png')
im_tr_1 = transform(im_1)
im_tr_1 = mx.nd.array(im_tr_1, mx.gpu(0))

im_2 = cv2.imread('/media/user/COMMON/lobanov/data/a7.#.rec.notrash/3.4/00041320.png')
im_tr_2 = transform(im_2)
im_tr_2 = mx.nd.array(im_tr_2, mx.gpu(0))


d1 = mxs.var('data_a')
emb1 = embedder(d1, '_a')
infer_shape(emb1, data_a=(1, 3, 32, 32), data=None)
emb1_arguments = emb1.list_arguments()
emb1_arguments.pop(0)
emb1_auxiliary = emb1.list_auxiliary_states()

mod = mx.module.Module(emb1, ['data_a'], context=[mx.gpu(0)])
mod.bind([('data_a', (1, 3, 32, 32))])
mod.init_params()
arg_params, aux_params = mod.get_params()

d2 = mxs.var('data_b')
emb2 = embedder(d2, '_b')
infer_shape(emb2, data_b=(1, 3, 32, 32), data=None)
emb2_arguments = emb2.list_arguments()
emb2_arguments.pop(0)
emb2_auxiliary = emb2.list_auxiliary_states()


shared_buffer = {}
for i, name in enumerate(emb1_arguments):
    shared_buffer[emb1_arguments[i]] = arg_params[name].as_in_context(mx.gpu(0))
    shared_buffer[emb2_arguments[i]] = arg_params[name].as_in_context(mx.gpu(0))

for i, name in enumerate(emb1_auxiliary):
    shared_buffer[emb1_auxiliary[i]] = aux_params[name].as_in_context(mx.gpu(0))
    shared_buffer[emb2_auxiliary[i]] = aux_params[name].as_in_context(mx.gpu(0))


distance = mxs.sqrt(mxs.sum(mxs.pow(emb1 - emb2, 2), axis=1))
infer_shape(distance, data_a=(1, 3, 32, 32), data_b=(1, 3, 32, 32), data=None)

siamese_exe = distance.simple_bind(mx.gpu(0), data_a=(1, 3, 32, 32), data_b=(1, 3, 32, 32), shared_buffer=shared_buffer)
distance_out = siamese_exe.forward(False, data_a=im_tr_1, data_b=im_tr_2)
print(distance_out)




