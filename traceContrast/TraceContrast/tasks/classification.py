import numpy as np
from . import _eval_protocols as eval_protocols
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
import scipy.io as scio
from sklearn.model_selection import cross_val_score
import torch

def eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='linear'):
    assert train_labels.ndim == 1 or train_labels.ndim == 2
    train_repr = model.encode(train_data, encoding_window='full_series' if train_labels.ndim == 1 else None)
    test_repr = model.encode(test_data, encoding_window='full_series' if train_labels.ndim == 1 else None)

    # train_repr = (train_repr-np.min(train_repr))/(np.max(train_repr)-np.min(train_repr))
    # test_repr = (test_repr-np.min(test_repr))/(np.max(test_repr)-np.min(test_repr))

    # scaler = StandardScaler()
    # train_repr = scaler.fit_transform(train_repr)
    # test_repr = scaler.fit_transform(test_repr)

    # print(str(np.mean(test_repr)))
    # print(str(np.std(test_repr)))

    if eval_protocol == 'linear':
        fit_clf = eval_protocols.fit_lr
    elif eval_protocol == 'svm':
        fit_clf = eval_protocols.fit_svm
    elif eval_protocol == 'knn':
        fit_clf = eval_protocols.fit_knn
    else:
        assert False, 'unknown evaluation protocol'

    def merge_dim01(array):
        return array.reshape(array.shape[0]*array.shape[1], *array.shape[2:])
    
    ### change
    # def merge_dim12(array):
    #     return array.reshape(array.shape[0], array.shape[1]*array.shape[2])

    # train_repr = merge_dim12(train_repr)
    # test_repr = merge_dim12(test_repr)
    # train_labels = train_labels.ravel()
    # test_labels = test_labels.ravel()
    ### change

    if train_labels.ndim == 2:
        train_repr = merge_dim01(train_repr)
        train_labels = merge_dim01(train_labels)
        test_repr = merge_dim01(test_repr)
        test_labels = merge_dim01(test_labels)

    clf = fit_clf(train_repr, train_labels)

    ##### TODO: cross validation
    # train_scores = -1 * cross_val_score(clf, train_repr, train_labels, cv=10, scoring='neg_mean_absolute_error')
    # test_scores = -1 * cross_val_score(clf, test_repr, test_labels, cv=10, scoring='neg_mean_absolute_error')
    # print("Train MAE scores: ",train_scores)
    # print("Test MAE scores: ",test_scores)
    #####

    train_acc = clf.score(train_repr,train_labels)  ###### add
    acc = clf.score(test_repr, test_labels)

    ##### TODO: predict and find wrong label period
    train_predict = clf.predict(train_repr)
    test_predict = clf.predict(test_repr)
    import numpy as np
    o1 = np.nonzero(train_predict-train_labels) # wrong train label
    o2 = np.nonzero(test_predict-test_labels) # wrong test label
    r1 = np.load('./train_real_label.npy')
    r2 = np.load('./test_real_label.npy')
    # print(r1[o1])
    # print(r2[o2])
    ### save wrong dff
    # scio.savemat(f'D:/lab/scn/3d/time_coding_2023_2_21/wrong/wrong_dff.mat',{'train_wrong':train_data[o1,:],'test_wrong':test_data[o2,:]})
    # scio.savemat(f'D:/lab/scn/3d/time_coding_2023_2_21/wrong/wrong_label.mat',{'train_wrong':r1[o1],'test_wrong':r2[o2]})
    #####

    print("Train ACC = "+str(train_acc))
    print("ACC = "+str(acc))
    # acc_str = str(acc)[2:5]
    # filename = f'D:/lab/scn/3d/time_coding_2023_2_21/time_code_emb_class2_{eval_protocol}_{acc_str}.mat'
    # scio.savemat(filename, {'train_emb':train_repr, 'test_emb':test_repr, 'train_labels':train_labels, 'test_labels':test_labels}) # save train_repr

    if eval_protocol == 'linear':
        y_score = clf.predict_proba(test_repr)
    else:
        y_score = clf.decision_function(test_repr)
    test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max()+1))
    # auprc = average_precision_score(test_labels_onehot, y_score[:,1]) # 2分类
    auprc = average_precision_score(test_labels_onehot, y_score) # n分类
    print("AUPRC = "+str(auprc))
    return y_score, { 'acc': acc, 'auprc': auprc }
