class DefaultConfigs(object):
    seed = 666
    # SGD
    weight_decay = 5e-4
    momentum = 0.9
    # learning rate
    init_lr = 0.0005
    # training parameters
    train_epoch = 100
    test_epoch = 1
    BATCH_SIZE_TRAIN = 32#%
    norm_flag = True
    gpus = '0'
    data = 'PaviaU'  # PaviaU-9-103-0.95 / Indian-16-200-0.9  / Houston2018-21  / Houston2013-15-0.95
    num_classes = 9
    patch_size = 11  # %
    num_tokens = 32  # 新增
    pca_components = 30  #
    d_state = 16  # 原16
    test_ratio = 0.98
    # model
    # model_type = 'Parallel MT'   # 'Parallel MT'  'Interval MT'  'Series MT'  'Series TM'  'Parallel Transformer-Mamba'  'Series Transformer-Mamba'  'Series Mamba-Transformer'
    # depth = 3
    embed_dim = 36

    ssm_ratio = 1
    pos = False
    cls = False

    # 新增ARConv参数
    arconv_inc = pca_components  # ARConv输入通道数
    arconv_outc = pca_components  # ARConv输出通道数
    arconv_kernel_size = 3  # 卷积核大小
    arconv_padding = 1  # 填充大小
    arconv_stride = 1  # 步长
    arconv_l_max = 9  # l_max参数
    arconv_w_max = 9  # w_max参数
    arconv_epoch_threshold = 100  # epoch阈值
    arconv_hw_range = [1, 63]  # hw_range参数
    use_arconv_residual = True  # 是否使用残差块
    num_arconv_blocks = 1  # 残差块数量

    # paths information
    checkpoint_path = ('./' + "checkpoint/" + data + '/' + 'TrainEpoch' + str(train_epoch) + '_TestEpoch' + str(
        test_epoch) + '_Batch' + str(BATCH_SIZE_TRAIN) \
                       + '/PatchSize' + str(patch_size) + '_TestRatio' + str(test_ratio) \
                       + '/' + '_embed' + str(embed_dim) + '_dstate' + str(d_state) + '_ratio' + str(ssm_ratio)
                       + '/' + '_hw_range' + str(arconv_hw_range) + '_epoch_threshold' + str(arconv_epoch_threshold))
    logs = checkpoint_path


config = DefaultConfigs()
