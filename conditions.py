from sympy.stats import (Expectation, Probability, Poisson, Binomial, NegativeBinomial, LogNormal, Weibull, Frechet,
                         Pareto, E, cdf, density, P)
from sympy import exp, Piecewise, Sum, symbols, summation, oo, integrate, gamma, log, diff, solve

def conditions_second_order_approximation(severity):
    # CONDITIONS FOR FIRST APPROXIMATION METHOD:
    cond1 = False
    cond2 = False
    cond3 = False
    cond4 = False
    cond5 = False
    cond6 = False

    # lim sup 1 condition:
    x, s = symbols('x s')
    function = density(severity)(x + s) / density(severity)(s)
    #lim_sup = function.limit_sup(function, s)
    #if(lim_sup == 1):
    #    cond1 = True

    # lim sup 2 condition
    #function2 = density(severity)(x * s) / density(s)
    #lim_sup2 = limit(function2, s)
    #if(lim_sup2 < oo):
    #    cond2 = True

    # Matuszewska index condition
    #function3 = log(lim_sup2) / log(x)
    #lim_sup3 = function3.limit(x, oo)
    #if(lim_sup3 < -1):
    #    cond3 = True

    # regularly varying condition cdf
    function4 = (1 - cdf(severity)(x*s))/(1 - cdf(severity)(s))
    lim_sup4 = function4.limit(s, oo)
    print(solve(lim_sup4))
    if(lim_sup4 > 0 and lim_sup4 < 1):
        cond4 = True
    elif(lim_sup4 == 1):
        cond5 = True

    # regularly varying condition pdf
    function4 = (density(severity)(x * s)) / (density(severity)(s))
    lim_sup4 = function4.limit(s, oo)
    if (lim_sup4 > 0 and lim_sup4 < 1):
        cond4 = True
    elif (lim_sup4 == 1):
        cond5 = True

    # regularly varying condition pdf
    if (cond1 and cond2 and cond3):
        return 1
    elif (cond4 and cond6):
        return 2
    elif (cond5 and cond6):
        return 3
    else:
        return 0
