import mxnet as mx
import numpy as np
import cPickle

def make_def_pairs(data, labels, num_pairs, image_shape): #эта фигня берёт наш CIFAR-10 и мирно делит его на пары, но как потом эти пары подавать на вход сетке и вообще, что происходит? 
    pairs = np.zeros((num_pairs, 2, image_shape[0], image_shape[1], # что такое data здесь? Это массивы значения пикселей? 
                     image_shape[2]))
    labels_pairs = np.zeros(num_pairs)
    num_same = 0 
    num_dif = 0
    k = 0 #! чем отличается k от num_pairs? k - счётчик, num_pairs - цель? 
    
    while k < num_pairs: # благодаря этому циклу мы вычленяем из нашего batch (или весь CIFAR-10) любые две пары с их labels, верно? 
        first_index = np.random.randint(data.shape[0]) #! data.shape[0] - зачем нам кол-во строк в матрице 32х32? 
        second_index = np.random.randint(data.shape[0]) #! что есть data.shape[0] из low, high, size? Бред какой-то.
        same = labels[first_index] == labels[second_index] # same is true, если картинки из одного класса
        if same and num_same < num_pairs/2: # второе условие - это проверка, что кол-во одинаковых картинок (случаев) не больше половины всех пар
            pairs[k,0] = data[first_index] #! что означает второй индекс? 2? 
            pairs[k,1] = data[second_index] 
            labels_pairs[k] = 1 # если картинки в паре равны, то окей, их лэйбл - 1
            num_same += 1
            k+=1
        elif not same and num_dif < num_pairs/2: 
            pairs[k,0]=data[first_index] 
            pairs[k,1]=data[second_index]
            labels_pairs[k]=0 # если картинки в паре разные, то окей, их лэйбл - 0
            num_dif += 1
            k+=1
    return pairs, labels_pairs  # pairs - индексы на пары, labels означает разные картинки в паре или нет, но как возвращать массивы image (пиксели, все дела)?
# зачем нам половина одинаковых пар, половина разных? 

class PairDataIter(mx.io.DataIter): # сюда нужно запихнуть make_def_makesъ
    
    def __init__(self, batch_size, mode='train'):
        #super(PairDataIter).__init__()
        assert mode in ['train', 'val']
        self.batch_size = batch_size
        self.provide_data = [('data_a', (batch_size, 3, 32, 32)), # предоставление двух бэтчей для выделения их них пар
                            ('data_b', (batch_size, 3, 32, 32))]
        # Нижние 4 строчки - бред? 
        self.data, self.label = self.load_cifar(path="/home/alex/Документы/formxnet/cifar-10-batches-py/") # а нужно ли здесь self слева
        #self.valid_data, self.valid_label = self.extractImagesAndLabels("/home/alex/Документы/formxnet/cifar-10-batches-py/","data_batch_1")
        
    
    def extractImagesAndLabels(self, path, file): # ну окей, я получил данные и лэйблы. Как теперь связать это с make_dif_pairs
         
        self.file = file
        f=open(path+file, 'rb')
        dict=cPickle.load(f) # загрузка в словарь
        images=dict['data']
        images = np.reshape(images,(10000, 3,32, 32)) 
        labels = dict['labels']
        imagearray = mx.nd.array(images) 
        labelarray = mx.nd.array(labels)
        return imagearray,labelarray # возвращаем массив картинок и меток (в виде значений пикселей) после десериализации
        
    def load_cifar(self, path): 
        training_data=[]
        training_label=[]
        for f in ("data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"):
            self.imgarray, self.lblarray = self.extractImagesAndLabels("/home/alex/Документы/formxnet/cifar-10-batches-py/", f)
            print "imgarray.size :"
            print self.imgarray.size              
            print "lblarray.size :"
            print self.lblarray.size
            if not training_data:
                print "not training_data"
                training_data = imgarray
                training_label = lblarray
                print "training_data after change"
                print training_data.shape
                print "training_label after change"
                print training_label.shape
            #else:  
             #   print "else"
                #training_data = mx.nd.concatenate([training_data, imgarray])
                #training_label = mx.nd.concatenate([training_label, lblarray])
              #  print "training_data_2 after change"
               # print training_data.shape
                #print "training_label_2 after change"
                #print training_label.shape
        return training_data, training_label # вроде то, что просили, а именно:  
    """
            далее нужно загрузить CIFAR-10 в виде 
            n - кол-во изображений в датасете
            data - список или массив всех изображений, shape (n, 3, 32, 32)
            label - номер класса для каждой картинки, shape (n, 1)
            Фактически нужно написать функцию load_cifar()
    """
   # def download_cifar10(self):
    #    data_dir="data"
     #   fnames = (os.path.join("/home/alex/", "cifar10_train.rec"),
      #        os.path.join("/home/alex/", "cifar10_val.rec"))
       # return fnames

    #if __name__ == '__main__':
     #   # download data
      #  self.train_fname, self.val_fname = self.download_cifar10()
    
    def next(self): # Здесь нужно сделать возвращение dataBatch, но чем они отличается от просто Batch? Гугл говорит, что это одно и то же
        pairs = make_def_pairs(self.data, self.label, self.batch_size, 
                                  (32, 32, 3))
        pass # зачем эта заглушка
        return mx.io.DataBatch(
            data=[mx.nd.array(np.moveaxis(pairs[0][:, 0], 3, 1)),
                 mx.nd.array(np.moveaxis(pairs[0][:, 1], 3, 1))],
            label = [mx.nd.array(pairs[1])],
            provide_data=self.provide_data,
            provide_label=self.provide_label        
            )
    
    # проверка работоспособности класса
    def saveCifarImage(self, array, path, file):
        self.array = array.asnumpy().transpose(1,2,0)
        # array is RGB. cv2 needs BGR
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        # save to PNG file
        return cv2.imwrite(path+file+".png", array)
     