import warnings


class DefaultConfig(object):
    env = 't4B'  # visdom env
    model = 'ACCNet'
    data = 'class2'  # cifar10 class1, class2
    from_scratch = False

    cifar10_dir = 'dataset/cifar10'  #'dataset/cifar10'
    class1_dir = 'dataset/class1'
    class2_dir = 'dataset/class2'


    load_model_path = None  # path of pretrained models
    # load_model_path = '/checkpoint/'+model+'_best.pth'
    save_model_path = 'checkpoint/'

    num_classes = 10
    train_batch_size = 64
    test_batch_size = 64
    num_workers = 4  # how many workers for loading data
    # print_freq = 20  # print info every N batch


    epoch = 100
    lr = 0.01
    momemtum = 0.9

    weight_decay = 1e-4
    T_max = 50


def parse(self, kwargs):
    '''
    update parser parameters, by dict 'kwargs'
    '''
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)

    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))


DefaultConfig.parse = parse
opt = DefaultConfig()
# opt.parse = parse