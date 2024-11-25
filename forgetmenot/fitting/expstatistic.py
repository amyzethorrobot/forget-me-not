import numpy as np
from .expfitting import exp_fitter2
from amylib.datablocks.defaultblocks import LossCurve

def treshold_stat(loss, th):
    
    net_num = loss.shape[0]
    th_indices = np.zeros(net_num, dtype = 'int')
    epoch_num = loss.shape[1]
    i = 0

    for l in loss:
        
        th_idc = np.argwhere(np.where(l <= th, 1, 0))
        
        if th_idc.shape[0] == 0:
            idx = epoch_num - 1 
        else:
            idx = th_idc[0]
            
        th_indices[i] = idx
        i +=1
        
    th_distr = np.zeros(epoch_num)
    for t in th_indices:
        th_distr[t - 1] += 1
        
    return th_indices, th_distr

def exp_statistic2(loss: LossCurve, 
                   time_th: int,
                   value_th: float):
    
    net_num = loss.series
    epoch_num = loss.length
    system_lr = loss.feature(name = 'lr')
    
    amps_array = []
    vels_array = []
    consts_array = []
    errors_array = []

    numbers = []

    th_indices = loss.treshold_points(value_th)

    start_point = time_th  
    end_point = (epoch_num - 1 - start_point)//4*3 + start_point
    endrel_point = end_point - start_point
    midrel_point = endrel_point//2

    approx_points = (0, midrel_point, midrel_point, endrel_point - 1)
    
    for i, l in enumerate(loss.data):
        
        if th_indices[i] >= time_th:
            continue
        else:

            approx_result = exp_fitter2(l[start_point:end_point], 
                                        np.arange(start_point, end_point, 1), 
                                        approx_points)
            
            #if not isinstance(approx_result, None):

            if True:

                (amp, vel, const, err) = approx_result

                if vel < 0 and const > 0 and amp > 0:

                    numbers.append(i)
                    amps_array.append(amp)
                    vels_array.append(vel)
                    consts_array.append(const)
                    errors_array.append(err)
            else:
                pass

    amps_array = np.array(amps_array)
    vels_array = np.array(vels_array)
    consts_array = np.array(consts_array)
    errors_array = np.array(errors_array)

    la_exp = np.exp(vels_array/2)
    la_max = (1 + la_exp)/ (2 * system_lr)
    la_min = (1 - la_exp)/ (2 * system_lr)

    return (amps_array, 
            vels_array, 
            consts_array, 
            la_max, 
            la_min, 
            errors_array, 
            numbers)