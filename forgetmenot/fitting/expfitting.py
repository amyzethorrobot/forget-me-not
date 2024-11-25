import numpy as np

def exp_fitter2(y: np.ndarray, 
                x: np.ndarray, 
                int_points: tuple[int, int, int, int]) -> tuple | None:

    '''
    Exponential fitting based on integral equations 
    (Original paper - EXPONENTIAL FITTING USING INTEGRAL EQUATIONS by E. Moore, 1974
    https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.1620080206)

    args:

    y: np.ndarray - input function values
    x: np.ndarray - input argument values
    int_points: tuple[int, int, int, int] - points 
    to set integral limits for I1 (int_om) and I2 (int_pn)

    return:

    tuple of amplitude (a), exp (l) and constant (b) for exponent a * exp(l * x) + b,
    mean relative error of approximation
    OR
    None if fitter can't perform an approximation
    '''

    np.seterr(all='raise')

    for i in range(3, 0, -1):
        if int_points[i] - int_points[i-1] < 0:
            raise ValueError('Invalid integral points {0} and {1}'.
                             format(i, i-1))        
    if int_points[3] == int_points[0]:
        raise ValueError('Empty approximation interval')   
    if int_points[3] == int_points[2]:
        raise ValueError('Empty interval of I2')
    if int_points[1] == int_points[0]:
        raise ValueError('Empty interval of I1')
    
    (o, m, p, n) = int_points
    
    int_om = np.trapz(y[o:m], x[o:m])
    int_pn = np.trapz(y[p:n], x[p:n])
    
    x_om = x[m] - x[o]
    x_pn = x[n] - x[p]
    y_om = y[m] - y[o]
    y_pn = y[n] - y[p]
    
    try:
        mu_coef = (y_om/int_om - y_pn/int_pn) / (x_om/int_om - x_pn/int_pn)
        l = (y_om - mu_coef * x_om) / int_om
        b = - mu_coef / l
    except ZeroDivisionError:
        return None

    try:
        exp_1 = np.sum(np.exp(l * x[o:n]))
        exp_2 = np.sum(np.exp(2 * l * x[o:n]))
        exp_y = np.sum(y[o:n] * np.exp(l * x[o:n]))
        a = (- b * exp_1 + exp_y)/exp_2
    except:
        return None

    test_exp = lambda x, amp, alph, cons: amp * np.exp(alph * x) + cons
    error = np.mean(np.abs(test_exp(x, a, l, b) - y)/y)
    
    return (a, l, b, error)