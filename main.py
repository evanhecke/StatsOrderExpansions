import math

from sympy.stats import Expectation, Probability, Poisson, Binomial, NegativeBinomial, LogNormal, Weibull, Frechet, Pareto, E, cdf, density, P, Normal
import matplotlib.pyplot as plt
import numpy as np
from math import comb
from sympy import exp, Piecewise, Sum, symbols, summation, oo, integrate, gamma, log, diff
import scipy.stats as stats


def initialize_distributions(claims_distribution_name, claims_params, severity_distribution_name, severity_params):
    # Initialize the claim amount distribution
    # And define the a and b value for the a,b,0-class
    if claims_distribution_name == 'Poisson':
        lambda_ = claims_params[0]
        N = Poisson('N', lambda_)
        a = 0
        b = lambda_
    elif claims_distribution_name == 'Binomial':
        n = int(claims_params[0])
        p = claims_params[1]
        N = Binomial('N', n, p)
        a = -p/(1-p)
        b = p*(n + 1)/(1-p)
    elif claims_distribution_name == 'Negative Binomial':
        r = claims_params[0]
        p = claims_params[1]
        N = NegativeBinomial('N', r, p)
        a = 1 - p
        b = (1 - p) * (r - 1)
    else:
        raise ValueError("Invalid claims distribution name")


    # Initialize the severity distribution
    if severity_distribution_name == 'Lognormal':
        mu = severity_params[0]
        sigma = severity_params[1]
        X = LogNormal('X', mu, sigma)
        exp_X = exp(mu + sigma**2 / 2)
    elif severity_distribution_name == 'Weibull':
        alpha = severity_params[0]
        beta = severity_params[1]
        X = Weibull('X', alpha, beta)
        exp_X = E(X)
    elif severity_distribution_name == 'Frechet':
        alpha = severity_params[0]
        beta = severity_params[1]
        X = Frechet('X', alpha, beta)
        exp_X = E(X)
    elif severity_distribution_name == 'Pareto':
        alpha = severity_params[0]
        xm = severity_params[1]
        X = Pareto('X', alpha, xm)
        if alpha == 1:
            exp_X = 1
        else:
            exp_X = E(X)
    else:
        raise ValueError("Invalid severity distribution name")

    return N, X, exp_X, a, b

def panjer_Recursion(amount, severity, s_values, a, b, delta):
    # should i add something to upper or lower bound for f????
    # and also check indices in bounds for g calculation
    f_lower = []
    f_upper = []
    g_lower = []
    g_upper = []

    # LOWER BOUNDS:
    for k in range(0, delta * int(max(s_values))):
        f_lower.append((cdf(severity)((k + 1) / delta) - cdf(severity)(k / delta)).evalf())

    # We first need to add the lowest value to lower bound of G, using convolution:
    conv = 0
    for i in range(0, E(amount) * 20):
        conv = conv + (density(amount)(i) * (f_lower[0] ** i)).evalf()
    g_lower.append(conv)

    for k in range(1, delta * int(max(s_values)) + 1):
        sum = 0
        for j in range(1, k + 1):
            sum = sum + ((1 / (1 - a * f_lower[0])) * (a + (b * j) / (k + 1)) * f_lower[(j - 1) * delta] * g_lower[(k - j) * delta])
        g_lower.append(sum)

    # UPPER BOUNDS:
    # first we have to add the lowest value to upper bound of G, using P(N=0):
    g_upper.append((density(amount)(0)).evalf())
    for k in range(1, delta * int(max(s_values)) + 2):
        f_upper.append((cdf(severity)(k / delta) - cdf(severity)((k - 1) / delta)).evalf())
        sum = 0
        for j in range(1, k + 1):
            sum = sum + ((a + (b * j) / k) * f_upper[(j-1) * delta] * g_upper[(k - j) * delta]).evalf()
        g_upper.append(sum)

    print(g_lower)
    print(g_upper)

    print(f_lower)
    print(f_upper)

def first_Order_Approximation(amount, severity, s_values):
    results = []
    for s in s_values:
        tailX = (1 - cdf(severity)(s)).evalf()
        result = E(amount) * tailX
        results.append(result)
    return results


'''def conditions_second_order_approximation(severity):
    # maybe i should check the conditions bh hand for the severity distributions,
    # this takes too much time
    # CONDITIONS FOR FIRST APPROXIMATION METHOD:
    cond1 = False
    cond2 = False
    cond3 = False
    cond4= False

    # lim sup 1 condition:
    x, s = symbols('x s')
    function = density(severity)(x + s) / density(severity)(s)
    lim_sup = function.limit(s, oo)
    if(lim_sup == 1):
        cond1 = True

    # lim sup 2 condition
    function2 = density(severity)(x * s) / density(s)
    lim_sup2 = function2.limit(s, oo)
    if(lim_sup2 < oo):
        cond2 = True

    # Matuszewska index condition
    function3 = log(lim_sup2) / log(x)
    lim_sup3 = function3.limit(x, oo)
    if(lim_sup3 < -1):
        cond3 = True

    function4 = (1 - cdf(severity)(x*s))/(1 - cdf(severity)(s))
    lim_sup4 = function4.limit(s, oo)

    if(cond1 and cond2 and cond3):
        return 1
    elif
    '''

def second_Order_Approximation(amount, severity, s_values, results1, exp_X):
    # check conditions to know which approximation
    y = symbols('y')
    exp_comb = E(amount * (amount - 1))
    '''results2_1 = []
    for s in s_values:
        eval_s = density(severity)(s).evalf()
        results2_1.append(exp_comb * exp_X * eval_s)
    # sum_results = [x + y for x, y in zip(results1, results2_1)]'''

    # for alpha between 0 and 1
    alpha = 0.5 # placeholder
    results2_2 = []
    for s in s_values:
        integr_s = integrate(1 - cdf(severity)(y), (y, 0, s))
        eval_s = (density(severity)(s)).evalf()
        expr = - (2 - alpha) * gamma(2 - alpha) / ((alpha - 1) * gamma(3 - 2 * alpha))
        results2_2.append(exp_comb * expr * eval_s * integr_s)
    sum_results = [x + y for x, y in zip(results1, results2_2)]

    # for pareto with alpha = 1
    '''results2_3 = []
    for s in s_values:
        integr_s = integrate(1 - cdf(severity)(y), (y, 0, s))
        eval_s = (density(severity)(s)).evalf()
        results2_3.append(exp_comb * eval_s * integr_s)
    sum_results = [x + y for x, y in zip(results1, results2_3)]'''

    return sum_results

def third_Order_Approximation(amount, severity, s_values, results_2):
    results3 = []
    x = symbols('x')
    density_function = density(severity)(x)
    derivative = diff(density_function, x)
    exp_operator = ((E(amount**2) - E(amount)) * E(severity**2) + (E(severity)**2) * (E(amount**3) - 3 * E(amount**2) + 2 * E(amount))).evalf()
    for s in s_values:
        eval_s = (derivative.subs(x, s)).evalf()
        results3.append(exp_operator * eval_s * 0.5)

    sum_results = [-z + y for z, y in zip(results_2, results3)]
    return sum_results


def doCalculations(claims_distribution_name, claims_params, severity_distribution_name, severity_params):
    # N, X = initialize_distributions(claims_distribution_name, claims_params, severity_distribution_name,
    #                                 severity_params)
    # print("Claim amount distribution:", cdf(N))
    # print("Severity distribution:", cdf(X))

    # print(Expectation(N).doit())
    # print((P(N > 4)).evalf(3))
    # Z = Normal('Z', 5, 2)
    # print(cdf(N)(1).evalf())
    # print(cdf(Z)(5).evalf())

    # Define range of s values
    s_min = 0
    s_max = 1001
    num_points = 10  # Number of points to plot
    s_values = np.linspace(s_min, s_max, num_points)

    # Initialize distributions
    N, X, exp_X, a ,b = initialize_distributions(claims_distribution_name, claims_params, severity_distribution_name,
                                    severity_params)

    # Evaluate first-order approximation
    results1 = first_Order_Approximation(N, X, s_values)


    # Evaluate second-order approximation
    results2 = second_Order_Approximation(N, X, s_values, results1, exp_X)

    # results3 = third_Order_Approximation(N, X, s_values, results2)

    # Plot the approximation results
    plt.plot(s_values, results1, linestyle='-', marker='', label='First-Order Approximation')
    plt.plot(s_values, results2, linestyle='-', marker='', label='Second-Order Approximation')
    # plt.plot(s_values, results3, linestyle='-', marker='', label='Third-Order Approximation')
    plt.xlabel('s values')
    plt.ylabel('Approximation')
    plt.title('Order Approximations')
    plt.legend()
    plt.show()

delta = 1
values = np.linspace(0, 5, 4)
panjer_Recursion(Poisson('N', 6), LogNormal('X', 1, 2), values, 0, 6, delta)