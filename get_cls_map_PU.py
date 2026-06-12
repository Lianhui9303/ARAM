import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
import matplotlib
import torch


#主要功能是将模型预测的类别标签（y_pred）映射到与真实类别标签图（y）相同大小的图像中，
# 生成一个分类结果图 cls_labels
# 只有真实类别标签不为零的像素位置才会被赋予预测的类别标签，而真实类别标签为零的像素位置则保持为零
def get_classification_map(y_pred, y):

    height = y.shape[0]
    width = y.shape[1]
    k = 0
    cls_labels = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            target = int(y[i, j])#获取当前像素的真实类别标签
            if target == 0:
                continue#真实类别标签为零，则跳过该像素，不进行任何操作
            else:
                cls_labels[i][j] = y_pred[k]+1
                k += 1#在下一次循环中访问 y_pred 中的下一个元素

    return  cls_labels
#返回生成的分类结果图 cls_labels，与y 大小相同,只包含预测的类别标签（真实类别标签不为零的位置）


#该函数的主要功能是将一个类别标签列表转换为对应的彩色图像映射
def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    #初始化输出数组y，大小为x_list的大小，每一行有三列零矩阵，代表一个像素的RGB值
    for index, item in enumerate(x_list):
        #index 是当前元素在列表中的位置，item 是该位置的类别标签。
        if item == 0:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 1:
            y[index] = np.array([147, 67, 46]) / 255.
        if item == 2:
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 3:
            y[index] = np.array([255, 100, 0]) / 255.
        if item == 4:
            y[index] = np.array([0, 255, 123]) / 255.
        if item == 5:
            y[index] = np.array([164, 75, 155]) / 255.
        if item == 6:
            y[index] = np.array([101, 174, 255]) / 255.
        if item == 7:
            y[index] = np.array([118, 254, 172]) / 255.
        if item == 8:
            y[index] = np.array([60, 91, 112]) / 255.
        if item == 9:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 10:
            y[index] = np.array([255, 255, 125]) / 255.
        if item == 11:
            y[index] = np.array([255, 0, 255]) / 255.
        if item == 12:
            y[index] = np.array([100, 0, 255]) / 255.
        if item == 13:
            y[index] = np.array([0, 172, 254]) / 255.
        if item == 14:
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 15:
            y[index] = np.array([171, 175, 80]) / 255.
        if item == 16:
            y[index] = np.array([101, 193, 60]) / 255.

    return y#函数返回矩阵 y，这个矩阵包含了每个类别标签对应的RGB颜色值



def classification_map(map, ground_truth, dpi, save_path):

    matplotlib.use('TkAgg')
    fig = plt.figure(frameon=False)#创建一个新的figure对象，frameon=False 表示不显示figure的边框
    fig.set_size_inches(ground_truth.shape[1]*2.0/dpi, ground_truth.shape[0]*2.0/dpi)
    #设置figure的大小，大小依据ground_truth图像的宽度和高度以及指定的dpi（每英寸点数）来计算

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    #创建一个新的轴（axis）对象，其范围覆盖整个figure。
    # 参数 [0., 0., 1., 1.] 表示左下角在（0,0），右上角在（1,1）的位置。
    ax.set_axis_off()#关闭轴的边框，使得图像看起来更干净
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    #分别隐藏x轴和y轴，使得坐标轴不显示在图像上
    fig.add_axes(ax)
    #将创建的轴对象添加到figure中

    ax.imshow(map)#imshow会将输入的二维数组数据映射为图像
    fig.savefig(save_path, dpi=dpi)

    return 0

def mytest(net, test_loader):
    count = 0
    net.eval()
    y_pred_test = 0
    y_test = 0
    for inputs, labels in test_loader:
        inputs = inputs.cuda()
        # #outputs, _ = net(inputs)
        # outputs = net(inputs)
        # outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        # 接收模型返回值
        result = net(inputs)

        # 处理不同返回值格式
        if isinstance(result, tuple):
            # 元组格式：取第一个元素（分类输出）
            outputs = result[0]
        elif isinstance(result, torch.Tensor):
            # 已经是张量
            outputs = result
        else:
            raise ValueError(f"模型返回了不支持的格式: {type(result)}")

        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))

    return y_pred_test, y_test


def get_cls_map(net, all_data_loader, y, save_path):

    y_pred, y_new = mytest(net, all_data_loader)
    cls_labels = get_classification_map(y_pred, y)
    x = np.ravel(cls_labels)
    gt = y.flatten()

    y_list = list_to_colormap(x)
    y_gt = list_to_colormap(gt)

    # y_re = np.reshape(y_list, (y.shape[0], y.shape[1], 3))
    # gt_re = np.reshape(y_gt, (y.shape[0], y.shape[1], 3))

    y_re = np.reshape(x, (y.shape[0], y.shape[1]))
    gt_re = np.reshape(gt, (y.shape[0], y.shape[1]))

    # save_dir = r"D:\image\MY\UP\train_0.02"
    # os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
    # save_path = os.path.join(save_dir, "UP0.97.mat")
    # sio.savemat(save_path, {'UP':y_re})

    # save_dir = r"D:\image\MY\Longkou\train_0.01"
    # os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
    # save_path = os.path.join(save_dir, "Longkou1.mat")
    # sio.savemat(save_path, {'lk':y_re})

    # save_dir = r"D:\image\GT\HZ\train_0.02"
    # os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
    # save_path = os.path.join(save_dir, "HZ.mat")
    # sio.savemat(save_path, {'HZ':gt_re})

    # classification_map(y_re, y, 300, save_path + 'PU_predictions.eps')
    # classification_map(y_re, y, 300, save_path + 'PU_SSPredictions.png')
    # classification_map(gt_re, y, 300, save_path + 'PU_gt.png')
    print('------Get classification maps successful-------')
