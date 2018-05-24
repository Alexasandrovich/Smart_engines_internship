# -*- coding: utf-8 -*-
import mxnet as mx
import numpy as np


def make_pairs_random(data, labels, num_pairs, image_shape):
    pairs = np.zeros((num_pairs, 2, image_shape[0], image_shape[1], image_shape[2]))
    labels_pairs = np.zeros(num_pairs)
    num_same = 0
    num_dif = 0
    k = 0

    while k < num_pairs:
        first_index = np.random.randint(data.shape[0])
        second_index = np.random.randint(data.shape[0])
        same = labels[first_index] == labels[second_index]
        if same and num_same < num_pairs/2:
            pairs[k, 0] = data[first_index]
            pairs[k, 1] = data[second_index]
            labels_pairs[k] = 1
            num_same += 1
            k += 1
        elif not same and num_dif < num_pairs/2:
            pairs[k, 0] = data[first_index]
            pairs[k, 1] = data[second_index]
            labels_pairs[k] = 0
            num_dif += 1
            k += 1
    # print("\n")
    return pairs, labels_pairs


class PairDataIter(mx.io.DataIter):
    def __init__(self, batch_size, mode='train'):
        super(PairDataIter, self).__init__()
        assert mode in ['train', 'val']
        self.batch_size = batch_size
        self.provide_label = [('label', (batch_size,)), ]
        self.provide_data = [('data_a', (batch_size, 3, 32, 32)), ('data_b', (batch_size, 3, 32, 32))]
        """
            далее нужно загрузить CIFAR-10 в виде 
            n - кол-во изображений в датасете
            data - список или массив всех изображений, shape (n, 3, 32, 32)
            label - номер класса для каждой картинки, shape (n, 1)
            Фактически нужно написать функцию load_cifar()
        """
        # self.data, self.label = load_cifar()

    def next(self):
        pairs = make_pairs_random(self.data, self.label, self.batch_size, (32, 32, 3))
        pass
        return mx.io.DataBatch(
            data=[mx.nd.array(np.moveaxis(pairs[0][:, 0], 3, 1)), mx.nd.array(np.moveaxis(pairs[0][:, 1], 3, 1))],
            label=[mx.nd.array(pairs[1])],
            provide_data=self.provide_data,
            provide_label=self.provide_label
        )

