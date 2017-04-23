__author__ = 'Created by foursking, edited by DPC'
from gen_feat import make_train_set
from gen_feat import make_test_set
from sklearn.model_selection import train_test_split
import xgboost as xgb
from gen_feat import report
from gen_feat import get_true


def xgboost_report_submission():
    train_start_date = '2016-03-08'
    train_end_date = '2016-04-09'
    result_start_date = '2016-04-09'
    result_end_date = '2016-04-14'

    valid_start_date = '2016-03-01'
    valid_end_date = '2016-04-02'
    valid_result_start_date = '2016-04-02'
    valid_result_end_date = '2016-04-07'

    test_start_date = '2016-03-15'
    test_end_date = '2016-04-16'

    user_index, training_data, label = make_train_set(train_start_date, train_end_date, result_start_date,
                                                      result_end_date)
    x_train, x_test, y_train, y_test = train_test_split(training_data.values,
                                                        label.values, test_size=0.2, random_state=0)
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)
    param = {'learning_rate': 0.1, 'n_estimators': 1000, 'max_depth': 3,
             'min_child_weight': 5, 'gamma': 0, 'subsample': 1.0, 'colsample_bytree': 0.8,
             'scale_pos_weight': 1, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic'}
    num_round = 283
    param['nthread'] = 4
    #param['eval_metric'] = "auc"
    plst = param.items()
    plst += [('eval_metric', 'logloss')]
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst = xgb.train(plst, dtrain, num_round, evallist)

    # Report with validation set
    valid_user_index, valid_trainning_date = make_test_set(valid_start_date, valid_end_date)
    valid_trainning_date = xgb.DMatrix(valid_trainning_date.values)
    pred_y = bst.predict(valid_trainning_date)

    valid_pred = valid_user_index.copy()
    valid_pred['label'] = pred_y
    valid_pred = valid_pred[valid_pred['label'] >= 0.014]
    valid_pred = valid_pred.sort_values('label', ascending=False).groupby('user_id').first().reset_index()
    valid_true = get_true(valid_result_start_date, valid_result_end_date)
    report(valid_pred, valid_true)

    valid_pred = valid_pred[valid_pred['label'] >= 0.016]
    print 0.016
    report(valid_pred, valid_true)

    valid_pred = valid_pred[valid_pred['label'] >= 0.018]
    print 0.018
    report(valid_pred, valid_true)

    valid_pred = valid_pred[valid_pred['label'] >= 0.02]
    print 0.02
    report(valid_pred, valid_true)

    valid_pred = valid_pred[valid_pred['label'] >= 0.022]
    print 0.022
    report(valid_pred, valid_true)

    valid_pred = valid_pred[valid_pred['label'] >= 0.024]
    print 0.024
    report(valid_pred, valid_true)

    valid_pred = valid_pred[valid_pred['label'] >= 0.026]
    print 0.026
    report(valid_pred, valid_true)

    valid_pred = valid_pred[valid_pred['label'] >= 0.028]
    print 0.028
    report(valid_pred, valid_true)

    valid_pred = valid_pred[valid_pred['label'] >= 0.03]
    print 0.03
    report(valid_pred, valid_true)

    # # Make submission for test set
    # test_user_index, test_trainning_data = make_test_set(test_start_date, test_end_date)
    # test_trainning_data = xgb.DMatrix(test_trainning_data.values)
    # pred_y = bst.predict(test_trainning_data)
    #
    # pred = test_user_index.copy()
    # pred['label'] = pred_y
    # pred = pred[pred['label'] >= 0.022]
    # pred = pred.sort_values('label', ascending=False).groupby('user_id').first().reset_index()
    # pred = pred[['user_id', 'sku_id']]
    # pred['user_id'] = pred['user_id'].astype(int)
    # pred.to_csv('./sub/submission.csv', index=False)


if __name__ == '__main__':
    xgboost_report_submission()
