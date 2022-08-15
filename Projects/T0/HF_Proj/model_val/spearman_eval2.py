import numpy as np
from scipy.stats import spearmanr
import os



def cal_spearman():
    file_path = 'prediction/param_2/'
    file_list = os.listdir(file_path)
    file_list.sort()
    spearman_list = []
    for f in file_list[1:]:
        arrs = np.load(file_path + f)
        y_predict = arrs['y_pred']
        y_true = arrs['y_true']
        for i in range(int(4800/200)):
            temp = []
            y_pred_sub = y_predict[:, i * 100 : (i + 1) * 100, 0].reshape((-1, 1))
            y_true_sub = y_true[:, i * 100 : (i + 1) * 100, 0].reshape((-1, 1))
            temp.append(spearmanr(y_true_sub, y_pred_sub)[0])
        spearman_list.append(np.mean(temp))
        # for i in range(4):
        #     y_pred_sub = y_predict[:,:,i].reshape((-1,1))
        #     y_true_sub = y_true[:, :, i].reshape((-1, 1))
        #     spearman_list.append(spearmanr(y_true_sub, y_pred_sub))
    print('here')

if __name__ == '__main__':
    cal_spearman()