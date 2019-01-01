#coding=utf-8
import os
import numpy as np
import tensorflow as tf
from numba import jit
from model import cdf_reg_model
from api.alpha_data_api import AlphaDataApi, generate_handles_by_dates

# generate_handles_by_dates(["20141231", "20151231", "20161231"], 1, 42, 5)中，各个参数的说明：
#   ["20141231", "20151231", "20161231"] ：各个数据集的截止日期，一般含有三个日期，这意味着：
#   "20141231"    ：训练集截止日期，规定不能超过20161231
#   "20151231"    ：交叉验证集截止日期，规定不能超过20161231
#   "20161231"    ：测试集截止日期，规定不能超过20161231
#    1 (c_class)  ：类别数。1代表回归，>1时代表分类
#   42 (n_train)  ：训练样本所用的历史天数。比如说，42代表使用42天历史的特征作为今天的总特征
#    5 (n_test)   ：预测的天数。比如说，5代表预测未来5天的总利润

# 指定使用哪张显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


def get_batch(handle, n_batch,is_seq=True):
    # batch_3d :
    #   1）回归问题：data = [ n_batch x n_train x (n_feature + 1) ]
    #   2）分类问题：(data, returns)，其中data即为回归问题中的data，returns则是对应的收益率
    #        之所以要返回收益率，是因为可能需要用到它来在交叉验证集上进行模型性能的评估
    # n_feature 代指特征数，目前固定为6。之所以上面写了 n_feature + 1，是因为最后一列是label
    # 如果是回归问题，label就是收益率；如果是分类问题，label就是收益率所应该划分到的类别
    batch_3d = handle.generate_batch(n_batch)
    # 将-1传入handle时，可以提取所有数据，同时还会将数据对应的ticker & dates（tnd）提取出来
    tnd = None
    if n_batch == -1:
        batch_3d, tnd = batch_3d
    # 如果是分类问题，不仅仅会把类别的label返回，同时还会返回对应的收益率（returns）
    if handle.n_class > 1:
        batch_3d, returns = batch_3d
    # n_train天中，第一天的label对应着当前batch的label
    features, labels = batch_3d[..., :-1], batch_3d[..., 0, -1]
    #对脏数据进行截断
    features = data_process(features)
    labels = labels[...,None]
    #is_seq==True的时候，返回序列模型的batch
    if is_seq== False: features = features.reshape([len(features), -1])  # 摊平以给全连接神经网络使用
    if tnd is None:
        return features, labels
    return features, labels, tnd

@jit
def data_process(features):
	#第5个feature是脏数据需要进行预处理，截断
	lower, upper = -1.0,1.0
	ill_fea = 4
	np.clip(features[:,:,ill_fea],lower,upper,out=features[:,:,ill_fea])
	return features



# 将模型打分输出到alpha csv
# xp：输入对应的placeholder
# sess：tensorflow的Session
# output：模型输出对应的tensor
# handle：数据生成器
def export_results(model, handle, name):
    # 取出handle对应的股票ticker以及日期（dates）
    x, y, tnd = get_batch(handle, -1)
    print("test x shape:{}".format(x.shape))
    handle_tickers, handle_dates = tnd.T
    # 建立ticker、日期字典
    tickers, dates = sorted(set(handle_tickers)), np.array(sorted(set(handle_dates)), np.str)
    tickers_dict = {ticker: i for i, ticker in enumerate(tickers)}
    dates_dict = {date: i for i, date in enumerate(dates)}
    # 建立结果表格
    base = np.full([len(dates_dict), len(tickers_dict)], 0.)

    print("Getting scores...")
    nend = n_batch = 512
    scores = np.array([])
    while nend < x.shape[0]:
        t_scores = model.predict(x[nend-n_batch : nend])
        scores = np.append(scores,t_scores)
        nend += n_batch
    t_scores = model.predict(x[nend-n_batch:])
    scores = np.append(scores,t_scores)
    print(t_scores.shape)
    print(scores.shape)

    print("Filling the form...")
    # 把分数填进结果表格
    target_base = base.copy()
    for date, row_num in dates_dict.items():
        mask = handle_dates == date
        local_tickers, local_scores, local_targets = handle_tickers[mask], scores[mask], y[mask][0]
        local_mask = [tickers_dict[ticker] for ticker in local_tickers if ticker in tickers_dict]
        base[row_num][local_mask] = local_scores.ravel()
        target_base[row_num][local_mask] = local_targets

    print("Transforming results to string...")
    # 把结果表格写进文件
    np_dates = np.array([
        "{}-{}-{}".format(date[:4], date[4:6], date[6:])
        for date in dates
    ]).reshape([-1, 1])
    body = np.hstack([np_dates, base.astype(np.str)])
    # targets代指ground truth，可以给大家一个近乎于“最完美的策略”所对应的alpha csv
    targets = np.hstack([np_dates, target_base.astype(np.str)])
    print("Writing results...")
    with open("results_{}.csv".format(name), "w") as file:
        file.write(",".join(["BusDay"] + tickers) + "\n")
        file.write("\n".join(",".join(line) for line in body))
    with open("targets_{}.csv".format(name), "w") as file:
        file.write(",".join(["BusDay"] + tickers) + "\n")
        file.write("\n".join(",".join(line) for line in targets))

    print("Done")



def run(n_class, batch=64, epoch=1e5, valid_batch=128, valid_epoch=1e4):
    epoch, valid_epoch = int(epoch), int(valid_epoch)
    train_handle, valid_handle, test_handle = generate_handles_by_dates(
        ["20141231", "20151231", "20161231"], n_class, 42, 5)
    n_data, n_train, n_feature = train_handle.shape
    input_dim, output_dim = n_feature, train_handle.n_class
    print("valid_shape",valid_handle.shape)

    model = cdf_reg_model(isTraining=True)

    print("training data : {}".format(n_data))
    for i in range(epoch):
        if i%valid_epoch == 0:
            valid_x,valid_y = get_batch(valid_handle,valid_batch)
            valid_loss = model.validate(valid_x,valid_y)
            print("valid loss {:2d} : {:8.6f}".format(i,valid_loss))
            model.save_model(epoch=i)
        train_x,train_y = get_batch(train_handle,batch)
        model.fit(train_x,train_y)

    #export_results(model,test_handle,name="t")
    export_results(model,valid_handle,name="v")

if __name__ == '__main__':
    run(1,batch=128,epoch=5e4,valid_batch=128,valid_epoch=1e3)  # 回归问题
