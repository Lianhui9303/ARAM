import torch
from models import ARAM#包含新改进mamba，以及卷积的完整模型

import os
from config_PU import config

os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus

import numpy as np
import scipy.io as sio #导入scipy.io，用于读取.mat格式的高光谱数据集
from sklearn.decomposition import PCA # 导入PCA类，用于高光谱数据降维
from sklearn.model_selection import train_test_split  # 导入数据集划分函数，用于训练/测试集拆分
# 导入分类评估指标（OA/AA/Kappa/混淆矩阵）
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score

import torch.nn as nn # 导入PyTorch神经网络模块，用于定义损失函数
import torch.optim as optim # 导入PyTorch优化器模块，用于模型参数优化
from operator import truediv # 导入除法函数，用于计算各类别准确率
import time # 导入时间库，用于统计训练/测试耗时

from utils.utils import Logger, mkdirs # 导入自定义工具：Logger（日志记录）、mkdirs（创建目录）
import get_cls_map_PU # 导入自定义工具，用于生成分类图（可视化结果）

def loadData():
    if config.data == 'PaviaU':#
        data = sio.loadmat('./data/PaviaU/PaviaU.mat')['paviaU'] # 读取高光谱数据（shape: H×W×C）
        labels = sio.loadmat('./data/PaviaU/PaviaU_gt.mat')['paviaU_gt'] # 读取标签数据（shape: H×W）

    elif config.data == 'xiongan':#
        # data = sio.loadmat('./data/xiongan/select_rs.mat')
        # print("Keys in the mat file:", data.keys())
        # labels = sio.loadmat('./data/xiongan/select_gt.mat')
        # print("Keys in the mat file:", labels.keys())
        data = sio.loadmat('./data/xiongan/select_rs.mat')['select_rs']
        labels = sio.loadmat('./data/xiongan/select_gt.mat')['select_gt']

    elif config.data == 'Trento':#
        # data = sio.loadmat('./data/Trento/HSI_Trento.mat')
        # print("Keys in the mat file:", data.keys())
        # labels = sio.loadmat('./data/Trento/GT_Trento.mat')
        # print("Keys in the mat file:", labels.keys())
        data = sio.loadmat('./data/Trento/HSI_Trento.mat')['HSI_Trento']
        labels = sio.loadmat('./data/Trento/GT_Trento.mat')['GT_Trento']

    elif config.data == 'Longkou':#
        data = sio.loadmat('./data/WHU-Hi-LongKou/WHU_Hi_LongKou.mat')['WHU_Hi_LongKou']
        labels = sio.loadmat('./data/WHU-Hi-LongKou/WHU_Hi_LongKou_gt.mat')['WHU_Hi_LongKou_gt']

    elif config.data == 'HZ':#
        #
        data = sio.loadmat('./data/Huzhou/research_corrected.mat')['research_corrected']
        labels = sio.loadmat('./data/Huzhou/research_gt.mat')['research_gt']

    elif config.data == 'zaoyuan':#
        data = sio.loadmat('./data/Zaoyuan/zaoyuan.mat')['HyperCube_OMIS4Class2']
        labels = sio.loadmat('./data/Zaoyuan/zaoyuan_gt.mat')['TruthMap2']

    elif config.data == 'SV':#
        data = sio.loadmat('./data/SV/Salinas.mat')['salinas_corrected']
        labels = sio.loadmat('./data/SV/Salinas_gt.mat')['salinas_gt']

    elif config.data == 'DTL':#
        data = sio.loadmat('./data/DTL/dth_new.mat')['dth_new']
        labels = sio.loadmat('./data/DTL/dth_gt_new.mat')['dth_gt_new']


    elif config.data == 'FangluTeaFarm':#
        data = sio.loadmat('./data/Fanglu Tea Farm/PHI_FangluTeaFarm.mat')['PHI_FangluTeaFarm']
        labels = sio.loadmat('./data/Fanglu Tea Farm/PHI_GroundTruthFanglu.mat')['PHI_GroundTruthFanglu']

    return data, labels

def applyPCA(X, numComponents):
    # Principal component analysis on HSI data（对高光谱数据执行PCA降维）
    newX = np.reshape(X, (-1, X.shape[2]))
    # 将3D数据（H×W×C）展平为2D（H*W×C），方便PCA计算
    pca = PCA(n_components=numComponents, whiten=True)
    # 初始化PCA：保留numComponents个主成分，白化（去相关性）
    newX = pca.fit_transform(newX)
    # 拟合并转换数据，完成降维
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    # 将2D数据恢复为3D（H×W×numComponents）
    return newX  # 返回降维后的数据

def padWithZeros(X, margin=2):
    #0填充，使图像块在进行卷积操作时都具有相同的输入尺寸（注释说明功能：边缘填充，避免裁剪Patch时边缘像素缺失）
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    # 创建填充后的空数组（边缘加margin行/列0）
    x_offset = margin  # x方向填充偏移量（margin）
    y_offset = margin  # y方向填充偏移量（margin）
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    # 将原始数据填入中间区域
    return newX  # 返回零填充后的数据

def createImageCubes(X, y, windowSize, removeZeroLabels = True):
    # 将高光谱图像（H×W×C）裁剪为大量 3D Patch
    margin = int((windowSize - 1) / 2)
    # 计算填充边缘宽度（Patch大小的一半，如15×15的margin=7）
    zeroPaddedX = padWithZeros(X, margin=margin)
    # 对降维后的数据执行零填充
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    # 存储所有Patch的数组（总数=H*W，每个Patch为windowSize×windowSize×C）
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    # 存储所有Patch的标签（每个Patch的标签=中心像素标签）
    patchIndex = 0  # Patch索引计数器
    for r in range(margin, zeroPaddedX.shape[0] - margin):  # 遍历填充后图像的行（跳过边缘margin）
        for c in range(margin, zeroPaddedX.shape[1] - margin):  # 遍历填充后图像的列（跳过边缘margin）
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]  # 裁剪当前位置的Patch（windowSize×windowSize）
            patchesData[patchIndex, :, :, :] = patch  # 存入Patch数组
            patchesLabels[patchIndex] = y[r-margin, c-margin]  # 存入Patch标签（原始图像对应位置的中心像素标签）
            patchIndex = patchIndex + 1  # 索引+1
    if removeZeroLabels:  # 是否移除零标签（未标注的像素）
        patchesData = patchesData[patchesLabels>0,:,:,:]  # 保留非零标签的Patch
        patchesLabels = patchesLabels[patchesLabels>0]  # 保留非零标签
        #patchesLabels -= 1  # 标签从0开始（适配PyTorch交叉熵损失函数的类别索引要求）
        # 重新映射标签为从0开始的连续整数
        unique_labels = np.unique(patchesLabels)
        print(f"非零标签值: {unique_labels}")

        # 创建映射字典
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}

        patchesLabels = np.array([label_mapping[old] for old in patchesLabels])  # 直接覆盖，而非新建变量

    print(f"处理后标签唯一值: {np.unique(patchesLabels)}")
    print(f"处理后标签范围: {np.min(patchesLabels)} 到 {np.max(patchesLabels)}")
    print(f"最终类别数: {len(np.unique(patchesLabels))}")
    return patchesData, patchesLabels  # 返回Patch数据和标签


def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=testRatio,random_state=randomState,stratify=y)

    return X_train, X_test, y_train, y_test

def create_data_loader():
    #构建数据加载器
    X, y = loadData()  # 加载原始数据和标签
    index = np.nonzero(y.reshape(y.shape[0]*y.shape[1]))
    # 获取非零标签像素的索引（用于后续分类图生成）
    index = index[0]  # 提取索引数组（展平后的一维索引）
    test_ratio = config.test_ratio  # 从配置文件读取测试集比例（如0.95，即5%训练、95%测试）
    patch_size = config.patch_size  # 从配置文件读取Patch大小（如15）
    pca_components = config.pca_components  # 从配置文件读取PCA降维后的波段数（如30）

    print('Hyperspectral data shape: ', X.shape)  # 打印原始数据形状（H×W×C）
    print('Label shape: ', y.shape)  # 打印标签形状（H×W）
    groundtruth = y  # 保存原始标签（用于分类图生成）

    print('\n... ... PCA tranformation ... ...')
    X_pca = applyPCA(X, numComponents=pca_components)  # 执行PCA降维
    print('Data shape after PCA: ', X_pca.shape)  # 打印降维后数据形状（H×W×pca_components）
    print('\n... ... create data cubes ... ...')
    X_pca, y = createImageCubes(X_pca, y, windowSize=patch_size)  # 生成3D Patch
    print('Data cube X shape: ', X_pca.shape)  # 打印Patch数据形状（N×windowSize×windowSize×pca_components，N为Patch总数）
    print('Data cube y shape: ', y.shape)  # 打印Patch标签形状（N）

    print('\n... ... create train & test data ... ...')
    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_pca, y, test_ratio)  # 划分训练/测试集
    print('Xtrain shape: ', Xtrain.shape)
    # 打印训练集Patch形状（N_train×windowSize×windowSize×pca_components）
    print('Xtest  shape: ', Xtest.shape)
    # 打印测试集Patch形状（N_test×windowSize×windowSize×pca_components）

    X = X_pca.reshape(-1, patch_size, patch_size, pca_components)  # 所有Patch添加通道维度（1），形状变为（N×windowSize×windowSize×pca_components×1）
    Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components)  # 训练集Patch添加通道维度
    Xtest = Xtest.reshape(-1, patch_size, patch_size, pca_components)  # 测试集Patch添加通道维度

    # 创建训练集、测试集、全数据集的Dataset实例
    X = TestDS(X, y)  # 全数据集（用于生成分类图）
    trainset = TrainDS(Xtrain, ytrain)  # 训练集
    testset = TestDS(Xtest, ytest)  # 测试集
    # 构建训练集数据加载器（批量加载、打乱）
    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=config.BATCH_SIZE_TRAIN,
                                               # 从配置文件读取训练批次大小（如64）
                                               shuffle=True,  # 训练集打乱
                                               drop_last=True
                                               # 丢弃最后不足一个批次的数据（避免批次大小不一致）
                                               )
    # 构建测试集数据加载器（不打乱、不丢弃）
    test_loader = torch.utils.data.DataLoader(dataset=testset,
                                               batch_size=config.BATCH_SIZE_TRAIN,
                                              # 测试批次大小=训练批次大小
                                               shuffle=False,  # 测试集不打乱
                                               num_workers=0,  # 单线程加载（避免多线程冲突）
                                               drop_last=False  # 不丢弃最后一个批次
                                              )
    # 构建全数据集数据加载器（用于生成分类图）
    all_data_loader = torch.utils.data.DataLoader(dataset=X,
                                               batch_size=config.BATCH_SIZE_TRAIN,
                                               shuffle=False,
                                               num_workers=0,
                                               drop_last=False
                                             )

    return train_loader, test_loader, y, index, all_data_loader, groundtruth  # 返回数据加载器、标签、索引、原始标签

""" Training dataset"""
class TrainDS(torch.utils.data.Dataset):
    def __init__(self, Xtrain, ytrain):
        self.len = Xtrain.shape[0]  # 训练集样本数（Patch总数）
        self.x_data = torch.FloatTensor(Xtrain)  # 将训练集数据转为FloatTensor（PyTorch模型输入格式）
        self.y_data = torch.LongTensor(ytrain)  # 将训练集标签转为LongTensor（分类任务标签格式）
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]  # 按索引返回单个样本（数据+标签）
    def __len__(self):
        return self.len  # 返回训练集样本数

""" Testing dataset"""
class TestDS(torch.utils.data.Dataset):
    def __init__(self, Xtest, ytest):
        self.len = Xtest.shape[0]  # 测试集/全数据集样本数
        self.x_data = torch.FloatTensor(Xtest)  # 数据转为FloatTensor
        self.y_data = torch.LongTensor(ytest)  # 标签转为LongTensor
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]  # 按索引返回单个样本（数据+标签）
    def __len__(self):
        return self.len  # 返回样本数

def train(train_loader):
    net = ARAM.AttentiveLayer(
        dim=config.pca_components,#输入数据的维度，这里使用PCA降维后的波段数
        d_state=config.d_state,#SSM（State Space Model）的状态维度，模型的核心参数之一
        input_resolution=(config.patch_size,config.patch_size),#输入图像块的分辨率，即高光谱图像中每个Patch的大小
        num_tokens=config.num_tokens,#模块中使用的令牌数量
        inner_rank=config.pca_components,#内部秩，通常与输入数据的维度相同
        mlp_ratio=2.0,#MLP（Multi-Layer Perceptron）的隐藏层与输入层的维度比
        num_classes=config.num_classes,#数据的类别数
        #新增的ARConv参数
        arconv_inc=config.arconv_inc,
        arconv_outc=config.arconv_outc,
        arconv_kernel_size=config.arconv_kernel_size,
        arconv_padding=config.arconv_padding,
        arconv_stride=config.arconv_stride,
        arconv_l_max=config.arconv_l_max,
        arconv_w_max=config.arconv_w_max,
        arconv_epoch_threshold=config.arconv_epoch_threshold,

        arconv_hw_range=config.arconv_hw_range,
        use_residual_block=config.use_arconv_residual,
        num_arconv_blocks=config.num_arconv_blocks


    ).cuda()#.cuda()方法用于将模型移动到GPU上进行加速计算

    criterion = nn.CrossEntropyLoss()  # 定义交叉熵损失函数（分类任务标准损失）
    optimizer = optim.Adam(net.parameters(), lr=0.001)  # 初始化Adam优化器来更新模型参数（学习率0.001）
    total_loss = 0  # 累计损失
    for epoch in range(config.train_epoch):  # 遍历训练轮次，每轮循环相当于一次完整的遍历整个训练数据集
        net.train()  # 模型设为训练模式（启用Dropout、BatchNorm更新）
        for i, (data, target) in enumerate(train_loader):  # 遍历训练集每个批次
            data, target = data.cuda(), target.cuda()
            # 数据和标签移至GPU（数据shape: [64, 1, 30, 15, 15]）
            outputs, _ = net(data,epoch=epoch)  # 模型前向传播，返回预测的logits和中间特征（outputs shape: [64, 9]）
            #outputs, _ = net(data)
            #outputs = net(data, epoch=epoch)
            loss = criterion(outputs, target)  # 计算损失（logits与真实标签的交叉熵）
            optimizer.zero_grad()  # 清空梯度（避免梯度累积）
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 优化器更新模型参数
            total_loss += loss.item()  # 每训练一个批次累计批次损失
        # 将当前轮次的平均损失写入日志
        log.write('[Epoch: %d]   [loss avg: %.4f] \n' % (epoch + 1, total_loss / (epoch + 1)))
    log.write('Finished Training')  # 训练完成，写入日志

    from thop import profile  # 导入thop库，用于计算模型参数量和FLOPs
    flops, params = profile(net, inputs=(data[0].unsqueeze(dim=0),))  # 计算参数量和FLOPs（输入单个样本）
    print('Params = ' + format(str(params / 1000 ** 2), '.6') + 'M')  # 打印参数量（单位：M）
    print('FLOPs = ' + format(str(flops / 1000 ** 3), '.6') + 'G')  # 打印FLOPs（单位：G）浮点运算次数
    return net, flops, params  # 返回训练好的模型、FLOPs、参数量

def mytest(net, test_loader):
    count = 0  # 计数器（用于初始化预测结果数组）
    net.eval()  # 模型设为评估模式（关闭Dropout、固定BatchNorm），
    # 这意味着在模型前向传播时，会关闭掉Dropout层，并且Batch Normalization层的参数会被固定，不会在评估过程中更新。
    y_pred_test = 0  # 存储所有测试集的预测结果
    y_test = 0  # 存储所有测试集的真实标签
    y_feature = 0  # 存储所有测试集的中间特征
    for inputs, labels in test_loader:  # 遍历测试集批次
        inputs = inputs.cuda()  # 测试数据移至GPU
        outputs, features = net(inputs)  # 对输入数据进行模型前向传播（返回logits和中间特征）
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)# logits转类别索引（CPU→numpy→取最大值索引）
        #detach()方法用于从计算图中分离outputs，防止对其进行梯度计算
        #outputs通常是一个包含每个类别得分（logits）的张量
        #将模型的预测输出从GPU移到CPU，并转换为numpy数组
        #然后使用np.argmax找到每个样本得分最高的类别作为预测类别

        features = features.detach().cpu().numpy()  # 中间特征转numpy数组
        if count == 0:  # 首次批次：初始化数组
            y_pred_test = outputs#初始化y_pred_test为当前批次的预测类别
            y_test = labels#初始化y_test为当前批次的真实类别
            y_feature = features#初始化y_feature为当前批次的中间特征
            count = 1#将计数器设置为1，表示已经处理了一个批次
        else:  # 非首次批次：拼接数组
            y_pred_test = np.concatenate((y_pred_test, outputs))  # 拼接预测结果
            y_test = np.concatenate((y_test, labels))  # 拼接真实标签
            y_feature = np.concatenate((y_feature, features))  # 拼接中间特征
    return y_pred_test, y_test, y_feature  # 返回预测结果、真实标签、中间特征

def AA_andEachClassAccuracy(confusion_matrix):
    #混淆矩阵储存的模型预测结果和真实标签
    #基于混淆矩阵计算每个类别的分类准确率（Each Class Accuracy）和平均准确率（AA）
    list_diag = np.diag(confusion_matrix)  # 提取混淆矩阵对角线元素（混淆矩阵对角线元素表示每个类别被正确分类的样本数量）
    list_raw_sum = np.sum(confusion_matrix, axis=1)  # 计算混淆矩阵每行和（每个类别的总样本数）
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))  # 计算各类别准确率（对角线/行和，处理除零错误）
    #np.nan_to_num() 函数用于将数组中的 NaN（非数字）值替换为0，以防止在某些类别没有出现样本时出现除零错误
    average_acc = np.mean(each_acc)  # 计算平均准确率（AA）
    return each_acc, average_acc  # 返回各类别准确率和AA

def acc_reports(y_test, y_pred_test):
    oa = accuracy_score(y_test, y_pred_test)  # 计算总体准确率（OA）
    confusion = confusion_matrix(y_test, y_pred_test)  # 计算混淆矩阵
    each_acc, aa = AA_andEachClassAccuracy(confusion)  # 计算各类别准确率和AA
    kappa = cohen_kappa_score(y_test, y_pred_test)  # 计算Kappa系数（衡量分类一致性）
    return oa*100, confusion, each_acc*100, aa*100, kappa*100
    # 返回OA、混淆矩阵、各类别准确率、AA、Kappa（均×100转为百分比）


if __name__ == '__main__':
    #主程序入口，这一行代码能够确保只有在直接运行脚本时才会执行以下代码，而不会在被导入为模块时执行
    mkdirs(config.checkpoint_path, config.checkpoint_path, config.logs)
    # 创建目录：模型保存路径、日志路径
    log = Logger()  # 初始化日志对象
    log.open(config.logs + config.data + '_log.txt', mode='a')  # 打开日志文件（追加模式）
    #mode='a' 表示以追加模式打开文件，这样可以避免覆盖之前的日志信息

    oa = []  # 存储多轮测试的OA
    acc = []  # 存储多轮测试的各类别准确率
    aa = []  # 存储多轮测试的AA
    kappa = []  # 存储多轮测试的Kappa
    train_times = []  # 存储每轮训练时间
    test_times = []  # 存储每轮测试时间
    for num in range(config.test_epoch):  # 遍历测试轮次（从配置文件读取，如5轮，重复5次执行训练和测试过程）
        # 构建数据加载器（每轮测试重新构建，确保数据划分一致）
        train_loader, test_loader, y_all, index, all_data_loader, y = create_data_loader()
        tic1 = time.perf_counter()  # 记录训练开始时间
        net, flops, params = train(train_loader)  # 训练模型
        toc1 = time.perf_counter()  # 记录训练结束时间
        current_train_time = toc1 - tic1
        train_times.append(current_train_time)  # 记录当前轮训练时间
        print("训练时间: {:.2f} 秒".format(current_train_time))  # 打印训练耗时

        tic2 = time.perf_counter()  # 记录测试开始时间
        y_pred_test, y_test, y_feature = mytest(net, test_loader)  # 测试模型
        toc2 = time.perf_counter()  # 记录测试结束时间
        current_test_time = toc2 - tic2
        test_times.append(current_test_time)  # 记录当前轮测试时间
        print("测试时间: {:.2f} 秒".format(current_test_time))  # 打印测试耗时

        # 计算评估指标
        each_oa, confusion, each_acc, each_aa, each_kappa = acc_reports(y_test, y_pred_test)
        oa.append(each_oa)  # 记录当前轮OA
        acc.append(each_acc)  # 记录当前轮各类别准确率
        aa.append(each_aa)  # 记录当前轮AA
        kappa.append(each_kappa)  # 记录当前轮Kappa
        # 将当前轮指标写入日志
        log.write('Test_Epoch: %.f, Each_OA: %.2f, Each_AA: %.2f, Each_kappa: %.2f , Train_Time: %.2fs, Test_Time: %.2fs'
                  '\n' % (num + 1, each_oa, each_aa, each_kappa, current_train_time, current_test_time))

        # 简化后的模型保存路径（解决Windows路径过长问题）
        save_path = os.path.join(
            config.checkpoint_path,  # 配置文件中的根路径（如./checkpoint/PaviaU/）
            f"T{num + 1}_OA{each_oa:.2f}"  # 简化目录名：测试轮次+OA（保留2位小数）
        )
        if not os.path.exists(save_path):  # 若目录不存在则创建
            os.makedirs(save_path)

        # 生成分类图（调用get_cls_map_PU工具，基于全数据集生成可视化分类结果）并保存到指定的路径中
        get_cls_map_PU.get_cls_map(net, all_data_loader, y, save_path)
        # 将当前轮评估指标写入txt文件（保存至模型目录）
        with open(save_path + "acc.txt", "w") as file:
            for item in each_acc:
                file.write("%s\n" % item)  # 写入各类别准确率
            file.write("%s\n" % each_oa)  # 写入OA
            file.write("%s\n" % each_aa)  # 写入AA
            file.write("%s\n" % each_kappa)  # 写入Kappa
            file.write("Train_Time: %.2f\n" % current_train_time)  # 写入当前轮训练时间
            file.write("Test_Time: %.2f\n" % current_test_time)  # 写入当前轮测试时间

    # 计算多轮测试的平均指标（均值、标准差、方差）并写入日志
    # 计算平均时间
    avg_train_time = np.mean(train_times)
    avg_test_time = np.mean(test_times)
    std_train_time = np.std(train_times)
    std_test_time = np.std(test_times)
    log.write(
        '   AVG:   OA: %.2f, std: %.2f, var: %.2f    AA: %.2f, std: %.2f, var: %.2f    Kappa: %.2f, std: %.2f, var: %.2f \n'
        % (np.mean(oa), np.std(oa), np.var(oa), np.mean(aa), np.std(aa), np.var(aa), np.mean(kappa), np.std(kappa),
           np.var(kappa)))
    # 写入平均时间到日志
    log.write('   TIME: Avg_Train: %.2f ± %.2fs, Avg_Test: %.2f ± %.2fs\n'
              % (avg_train_time, std_train_time, avg_test_time, std_test_time))
    # 将平均指标保存至独立txt文件（方便后续对比）
    with open((config.logs + 'AVG_OA%.3f_AA%.3f_Kappa%.3f.txt' % (np.mean(oa), np.mean(aa), np.mean(kappa))),
              'w') as file:
        file.write(
            f'Acc per class = {np.round(np.mean(acc, 0), decimals=2)} +- {np.round(np.std(acc, 0), decimals=2)}\n')
        file.write("OA: %.2f, std: %.2f, var: %.2f\n" % (np.mean(oa), np.std(oa), np.var(oa)))
        file.write("AA: %.2f, std: %.2f, var: %.2f\n" % (np.mean(aa), np.std(aa), np.var(aa)))
        file.write("Kappa: %.2f, std: %.2f, var: %.2f\n" % (np.mean(kappa), np.std(kappa), np.var(kappa)))
        file.write("Params: %.4f\n" % (params / 1000 ** 2))  # 写入参数量
        file.write("Flops: %.4f\n" % (flops / 1000 ** 3))  # 写入FLOPs
        file.write("=== 时间统计 ===\n")
        file.write("平均训练时间: %.2f ± %.2f 秒\n" % (avg_train_time, std_train_time))
        file.write("平均测试时间: %.2f ± %.2f 秒\n" % (avg_test_time, std_test_time))
