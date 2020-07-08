# Tianci-O2O-coupon-usage-predict
The first time to participate in machine learning competitions

赛事地址[天池新人实战赛o2o优惠券使用预测](https://tianchi.aliyun.com/competition/entrance/231593/information)

## 数据
  本赛题提供用户在2016年1月1日至2016年6月30日之间真实线上线下消费行为，预测用户在2016年7月领取优惠券后15天以内的使用情况。
注意： 为了保护用户和商家的隐私，所有数据均作匿名处理，同时采用了有偏采样和必要过滤。

## 评价方式
  本赛题目标是预测投放的优惠券是否核销。针对此任务及一些相关背景知识，使用优惠券核销预测的平均AUC（ROC曲线下面积）作为评价标准。 即对每个优惠券coupon_id单独计算核销预测的AUC值，再对所有优惠券的AUC值求平均作为最终的评价标准。
  
## 字段表
Table 1: 用户线下消费和优惠券领取行为
|Field|Description|
| :----:| :----: |
|User_id|用户ID|
|Merchant_id|商户ID|
|Coupon_id|优惠券ID：null表示无优惠券消费，此时Discount_rate和Date_received字段无意义|
|Discount_rate|优惠率：x \in [0,1]代表折扣率；x:y表示满x减y。单位是元|
|Distance|user经常活动的地点离该merchant的最近门店距离是x*500米（如果是连锁店，则取最近的一家门店），x\in[0,10]；null表示无此信息，0表示低于500米，10表示大于5公里；|
|Date_received|领取优惠券日期|
|Date|消费日期：如果Date=null & Coupon_id != null，该记录表示领取优惠券但没有使用，即负样本；如果Date!=null & Coupon_id = null，则表示普通消费日期；如果Date!=null & Coupon_id != null，则表示用优惠券消费日期，即正样本；|

Table 2: 用户线上点击/消费和优惠券领取行为
|Field|Description|
| :----:| :----: |
|User_id|用户ID|
|Merchant_id|商户ID|
|Action|0 点击， 1购买，2领取优惠券|
|Coupon_id|优惠券ID：null表示无优惠券消费，此时Discount_rate和Date_received字段无意义|
|Discount_rate|优惠率：x \in [0,1]代表折扣率；x:y表示满x减y。单位是元|
|Date_received|领取优惠券日期|
|Date|消费日期：如果Date=null & Coupon_id != null，该记录表示领取优惠券但没有使用，即负样本；如果Date!=null & Coupon_id = null，则表示普通消费日期；如果Date!=null & Coupon_id != null，则表示用优惠券消费日期，即正样本；|

Table 3：用户O2O线下优惠券使用预测样本
|Field|Description|
| :----:| :----: |
|User_id|用户ID|
|Merchant_id|商户ID|
|Coupon_id|优惠券ID：null表示无优惠券消费，此时Discount_rate和Date_received字段无意义|
|Discount_rate|优惠率：x \in [0,1]代表折扣率；x:y表示满x减y。单位是元|
|Distance|user经常活动的地点离该merchant的最近门店距离是x*500米（如果是连锁店，则取最近的一家门店），x\in[0,10]；null表示无此信息，0表示低于500米，10表示大于5公里；|
|Date_received|领取优惠券日期|

Table 4：选手提交文件字段，其中user_id,coupon_id和date_received均来自Table 3,而Probability为预测值
|Field|Description|
| :----:| :----: |
|User_id|用户ID|
|Coupon_id|优惠券ID|
|Date_received|领取优惠券日期|
|Probability|15天内用券概率，由参赛选手给出|

## 赛题分析
  提供数据的区间是2016-01-01~2016-06-30，预测七月份用户领券使用情况，即用或者不用。转化为二分类问题，所以适用于分类模型，例如GBDT，RF，XGBoost等。
  
## 数据划分
  在数据集的划分方面借鉴了比赛第一名[wepe](https://github.com/wepe/O2O-Coupon-Usage-Forecast)的策略。
  可以采用滑窗的方法得到多份训练数据集，特征区间越小，得到的训练数据集越多。以下是一种划分方式：  
  ||预测区间（提取label）|特征区间（提取feature）|
  |:---:|:---:|:---:|
  |测试机|20160701~20160731|20160101~20160630|
  |训练集1|20160515~20160615|20160201~20160514|
  |训练集2|20160414~20160514|20160101~20160414|
  
  划取多份训练集，一方面可以增加训练样本，另一方面可以做交叉验证实验，方便调参。
  
## 特征工程
  本次赛事提供了用户线上和线下消费、领取优惠券两个数据集。根据这两份数据集，我们可以把特征主要分为5个方面：
    1.用户特征
    2.商户特征
    3.优惠券特征
    4.用户商户组合特征
    5.用户优惠券组合特征
 
 ## 算法
 使用了在比赛中使用最广泛的XGBoost模型，使用模型自带CV进行参数调优，单模型下获得0.8的AUC,代码如下：

```Python
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
      xgb.callback.early_stop(40)])
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
  ```
  
  运行结果：
  ![](https://github.com/kilone/Tianci-O2O-coupon-usage-predict/blob/master/Data/%E6%89%B9%E6%B3%A8%202020-07-03%20214842.png)
  
  ## 最终成绩
  ![](https://github.com/kilone/Tianci-O2O-coupon-usage-predict/blob/master/Data/final%20score.png)
   
  
  

