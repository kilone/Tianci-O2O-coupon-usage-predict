import datetime
import os

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import auc, roc_curve


def get_processed_data():
    dataset1 = pd.read_csv('data_preprocessed_3/ProcessDataSet1.csv')
    dataset2 = pd.read_csv('data_preprocessed_3/ProcessDataSet2.csv')
    dataset3 = pd.read_csv('data_preprocessed_3/ProcessDataSet3.csv')

    dataset1.drop_duplicates(inplace=True)
    dataset2.drop_duplicates(inplace=True)
    dataset3.drop_duplicates(inplace=True)

    dataset12 = pd.concat([dataset1, dataset2], axis=0)

    dataset12.fillna(0, inplace=True)
    dataset3.fillna(0, inplace=True)

    return dataset12, dataset3


def train_xgb(dataset12, dataset3):
    predict_dataset = dataset3[['User_id', 'Coupon_id', 'Date_received']].copy()
    predict_dataset.Date_received = pd.to_datetime(predict_dataset.Date_received, format='%Y-%m-%d')
    predict_dataset.Date_received = predict_dataset.Date_received.dt.strftime('%Y%m%d')

    # 将数据转化为dmatric格式
    dataset12_x = dataset12.drop(
        columns=['User_id', 'Merchant_id', 'Discount_rate', 'Date_received', 'discount_rate_x', 'discount_rate_y',
                 'Date', 'Coupon_id', 'label'], axis=1)
    dataset3_x = dataset3.drop(
        columns=['User_id', 'Merchant_id', 'Discount_rate', 'Date_received', 'discount_rate_x', 'discount_rate_y',
                 'Coupon_id'], axis=1)

    train_dmatrix = xgb.DMatrix(dataset12_x, label=dataset12.label)
    predict_dmatrix = xgb.DMatrix(dataset3_x)

    # xgboost模型训练
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'gamma': 0.1,
              'min_child_weight': 1.1,
              'max_depth': 5,
              'lambda': 10,
              'subsample': 0.7,
              'colsample_bytree': 0.7,
              'colsample_bylevel': 0.7,
              'eta': 0.02,
              # 'tree_method': 'gpu_hist',
              # 'gpu_id': '1',
              # 'n_gpus': '-1',
              'seed': 0,
              'nthread': cpu_jobs,
              # 'predictor': 'gpu_predictor'
              }

    # 使用xgb.cv优化num_boost_round参数
    cvresult = xgb.cv(params, train_dmatrix, num_boost_round=10000, nfold=2, metrics='auc', seed=0, callbacks=[
        xgb.callback.print_evaluation(show_stdv=False),
        xgb.callback.early_stop(40)
    ])
    num_round_best = cvresult.shape[0] - 1
    print('Best round num: ', num_round_best)

    # 使用优化后的num_boost_round参数训练模型
    watchlist = [(train_dmatrix, 'train')]
    model = xgb.train(params, train_dmatrix, num_boost_round=num_round_best, evals=watchlist)

    model.save_model('train_dir_2/xgbmodel4')
    params['predictor'] = 'cpu_predictor'
    model = xgb.Booster(params)
    model.load_model('train_dir_2/xgbmodel4')

    # predict test set
    dataset3_predict = predict_dataset.copy()
    dataset3_predict['label'] = model.predict(predict_dmatrix)

    # 标签归一化
    dataset3_predict.label = MinMaxScaler(copy=True, feature_range=(0, 1)).fit_transform(
        dataset3_predict.label.values.reshape(-1, 1))
    dataset3_predict.sort_values(by=['Coupon_id', 'label'], inplace=True)
    dataset3_predict.to_csv("train_dir_2/xgb_preds_6.csv", index=None, header=None)
    print(dataset3_predict.describe())

    # 在dataset12上计算auc
    # model = xgb.Booster()
    # model.load_model('train_dir_2/xgbmodel')

    temp = dataset12[['Coupon_id', 'label']].copy()
    temp['pred'] = model.predict(xgb.DMatrix(dataset12_x))
    temp.pred = MinMaxScaler(copy=True, feature_range=(0, 1)).fit_transform(temp['pred'].values.reshape(-1, 1))
    print(myauc(temp))


# 性能评价函数
def myauc(test):
    testgroup = test.groupby(['Coupon_id'])
    aucs = []
    for i in testgroup:
        tmpdf = i[1]
        if len(tmpdf['label'].unique()) != 2:
            continue
        fpr, tpr, thresholds = roc_curve(tmpdf['label'], tmpdf['pred'], pos_label=1)
        aucs.append(auc(fpr, tpr))
    return np.average(aucs)


if __name__ == '__main__':
    start = datetime.datetime.now()
    print(start.strftime('%Y-%m-%d %H:%M:%S'))
    # log = '%s\n' % start.strftime('%Y-%m-%d %H:%M:%S')
    cpu_jobs = os.cpu_count() - 1
    date_null = pd.to_datetime('1970-01-01', format='%Y-%m-%d')

    dataset12, dataset3 = get_processed_data()
    # analysis()
    # detect_duplicate_columns()
    # feature_importance_score()

    # grid_search_gbdt()
    # train_gbdt()
    # predict('gbdt')

    # grid_search_xgb()
    train_xgb(dataset12, dataset3)

    # print('predict: start predicting......')
    # # predict('xgb')
    # print('predict: predicting finished.')

    # log += 'time: %s\n' % str((datetime.datetime.now() - start)).split('.')[0]
    # log += '----------------------------------------------------\n'
    # open('%s.log' % os.path.basename(__file__), 'a').write(log)
    # print(log)
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('time costed is: %s s' % (datetime.datetime.now() - start).seconds)