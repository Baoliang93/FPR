"""
Some Useful Functions and Classes
"""

import shutil
from abc import ABCMeta, abstractmethod
from threading import Lock
from sys import stdout
import torch
import torch.nn as nn
import numpy as np
import scipy
from scipy import stats


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return


    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss



class AverageMeter:
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


"""
Metrics for IQA performance
-----------------------------------------

Including classes:
    * Metric (base)
    * MAE
    * SROCC
    * PLCC
    * RMSE

"""

class Metric(metaclass=ABCMeta):
    def __init__(self):
        super(Metric, self).__init__()
        self.reset()
        self.scale = 100.0
    
    def reset(self):
        self.x1 = []
        self.x2 = []

    @abstractmethod
    def _compute(self, x1, x2):
        return

    def logistic(self, X, beta1, beta2, beta3, beta4, beta5):
        logistic_part = 0.5 - 1./(1 + np.exp(beta2 * (X - beta3)))
        yhat = beta1 * logistic_part + beta4 * X + beta5
        return yhat

    def compute(self):
        mos = np.array(self.x1, dtype=np.float).flatten()/self.scale 
        obj_score = np.array(self.x2, dtype=np.float).flatten()/self.scale 
        beta1 = np.max(mos)
        beta2 = np.min(mos)
        beta3 = np.mean(obj_score)
        beta = [beta1, beta2, beta3, 0.1, 0.1]  # inital guess for non-linear fitting

        fit_stat = ''
        try:
            popt, _ = scipy.optimize.curve_fit(self.logistic, xdata=obj_score, ydata=mos, p0=beta, maxfev=10000)
        except:
            popt = beta
            fit_stat = '[nonlinear reg failed]'
        ypred = self.logistic(obj_score, popt[0], popt[1], popt[2], popt[3], popt[4])

        # print('mos:', mos[1:10])
        # print('ypred:', ypred[1:10])
        mos, ypred = mos*self.scale, ypred*self.scale 
        return self._compute(mos.ravel(), ypred.ravel())

    def _check_type(self, x):
        return isinstance(x, (float, int, np.ndarray))

    def update(self, x1, x2):
        if self._check_type(x1) and self._check_type(x2):
            self.x1.append(x1)
            self.x2.append(x2)
        else:
            raise TypeError('Data types not supported')

class MAE(Metric):
    def __init__(self):
        super(MAE, self).__init__()

    def _compute(self, x1, x2):
        return np.sum(np.abs(x2-x1))

class SROCC(Metric):
    def __init__(self):
        super(SROCC, self).__init__()
    
    def _compute(self, x1, x2):
        return stats.spearmanr(x1, x2)[0]

class PLCC(Metric):
    def __init__(self):
        super(PLCC, self).__init__()

    def _compute(self, x1, x2):
        return stats.pearsonr(x1, x2)[0]

class RMSE(Metric):
    def __init__(self):
        super(RMSE, self).__init__()

    def _compute(self, x1, x2):
        return np.sqrt(((x2 - x1) ** 2).mean())


def limited_instances(n):
    def decorator(cls):
        _instances = [None]*n
        _lock = Lock()
        def wrapper(idx, *args, **kwargs):
            nonlocal _instances
            with _lock:
                if idx < n:
                    if _instances[idx] is None: _instances[idx] = cls(*args, **kwargs)   
                else:
                    raise KeyError('index exceeds maximum number of instances')
                return _instances[idx]
        return wrapper
    return decorator


class SimpleProgressBar:
    def __init__(self, total_len, pat='#', show_step=False, print_freq=1):
        self.len = total_len
        self.pat = pat
        self.show_step = show_step
        self.print_freq = print_freq
        self.out_stream = stdout

    def show(self, cur, desc):
        bar_len, _ = shutil.get_terminal_size()
        # The tab between desc and the progress bar should be counted.
        # And the '|'s on both ends be counted, too
        bar_len = bar_len - self.len_with_tabs(desc+'\t') - 2
        bar_len = int(bar_len*0.8)
        cur_pos = int(((cur+1)/self.len)*bar_len)
        cur_bar = '|'+self.pat*cur_pos+' '*(bar_len-cur_pos)+'|'

        disp_str = "{0}\t{1}".format(desc, cur_bar)

        # Clean
        self.write('\033[K')

        if self.show_step and (cur % self.print_freq) == 0:
            self.write(disp_str, new_line=True)
            return

        if (cur+1) < self.len:
            self.write(disp_str)
        else:
            self.write(disp_str, new_line=True)

        self.out_stream.flush()

    @staticmethod
    def len_with_tabs(s):
        return len(s.expandtabs())

    def write(self, content, new_line=False):
        end = '\n' if new_line else '\r'
        self.out_stream.write(content+end)
