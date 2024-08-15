from math import gamma

from sympy.stats import (Expectation, Probability, Poisson, Binomial, NegativeBinomial, LogNormal,
                         Weibull, Frechet, Pareto, E, cdf, density, P, Normal)
from sympy import exp, symbols, integrate, oo
import numpy as np
import matplotlib.pyplot as plt
import conditions

def initialize_distributions(claims_distribution_name, claims_params, severity_distribution_name, severity_params):
    # Initialize the claim amount distribution
    # And define the a and b values from the a,b,0-class
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
        r = int(claims_params[0])
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
        lambda_ = severity_params[0]
        k = severity_params[1]
        X = Weibull('X', lambda_, k)
        exp_X = E(X)
    elif severity_distribution_name == 'Frechet':
        alpha = severity_params[0]
        beta = severity_params[1]
        min = severity_params[2]
        X = Frechet('X', alpha, beta, min)
        exp_X = E(X)
    elif severity_distribution_name == 'Pareto':
        alpha = severity_params[0]
        xm = severity_params[1]
        X = Pareto('X', xm, alpha)
        if alpha == 1:
            exp_X = 1
        elif alpha < 1:
            exp_X = oo
        else:
            exp_X = E(X)
    else:
        raise ValueError("Invalid severity distribution name")

    return N, X, exp_X, a, b

def first_Order_Approximation(amount, severity, s_values):
    results = []
    for s in s_values:
        tailX = (1 - cdf(severity)(s)).evalf()
        result = E(amount) * tailX
        results.append(result)
    return results


def second_Order_Approximation(amount, severity, severity_distribution_name, severity_params, s_values, results1, exp_X):
    # Create empty array for the computed second order approximation values
    results2 = []
    sum_results2 = []

    # Create symbol to be used in density functions of sympy package
    y = symbols('y')

    # Calculate the expected value term of combinator found in all second order equations
    exp_comb = E(amount * (amount - 1))

    # Correctly identify which approximation to use and calculate it
    if severity_distribution_name in ["Pareto", "Frechet"]:
        # Pareto and Frechet have a distinct asymptotic approximation
        alpha = severity_params[0]
        if severity_distribution_name == "Pareto":
            min = severity_params[1]
        else:
            min = severity_params[2]
        if alpha == 1:
            for s in s_values:
                integr_s = integrate(1 - cdf(severity)(y), (y, min, s))
                eval_s = (density(severity)(s)).evalf()
                results2.append(2 * exp_comb * eval_s * integr_s)
            sum_results2 = [x + y for x, y in zip(results1, results2)]
        elif alpha > 0 and alpha < 1:
            for s in s_values:
                integr_s = integrate(1 - cdf(severity)(y), (y, min, s))
                eval_s = (density(severity)(s)).evalf()
                expr = - (2 - alpha) * gamma(2 - alpha) / ((alpha - 1) * gamma(3 - 2 * alpha))
                results2.append(exp_comb * expr * eval_s * integr_s)
            sum_results2 = [x + y for x, y in zip(results1, results2)]
        else:
            # placeholder for cases with alpha > 1
            for s in s_values:
                eval_s = density(severity)(s).evalf()
                results2.append(2 * exp_comb * exp_X * eval_s)
            sum_results2 = [x + y for x, y in zip(results1, results2)]
    else:
        #placeholder for second order calculation for other cases of frechet and pareto
        # The asymptotic approximation is valid for Lognormal distribution and heavy-tailed Weibull distribution
        for s in s_values:
            eval_s = density(severity)(s).evalf()
            results2.append(2 * exp_comb * exp_X * eval_s)
        sum_results2 = [x + y for x, y in zip(results1, results2)]

    return sum_results2

# def higher_Order_Approximations(amount, severity, severity_distribution_name, severity_params, s_values, results1, exp_X):



def doCalculations(claims_distribution_name, claims_params, severity_distribution_name, severity_params):

    # print(Expectation(N).doit())
    # print((P(N > 4)).evalf(3))
    # Z = Normal('Z', 5, 2)
    # print(cdf(N)(1).evalf())
    # print(cdf(Z)(5).evalf())

    # Define range of s values
    if severity_distribution_name == "Pareto":
        s_min = severity_params[1]
    elif severity_distribution_name == "Frechet":
        s_min = severity_params[2]
    else:
        s_min = 1
    s_max = 1001
    num_points = 101  # Number of points to plot
    s_values = np.linspace(s_min, s_max, num_points)

    # Initialize distributions
    N, X, exp_X, a, b = initialize_distributions(claims_distribution_name, claims_params, severity_distribution_name,
                                    severity_params)
    print("Claim amount distribution:", cdf(N))
    print("Severity distribution:", cdf(X))

    # Evaluate first-order approximation
    results1 = first_Order_Approximation(N, X, s_values)

    # conditions.conditions_second_order_approximation(X)

    # Evaluate second-order approximation
    results2 = second_Order_Approximation(N, X, severity_distribution_name, severity_params, s_values, results1, exp_X)

    # results3 = third_Order_Approximation(N, X, s_values, results2)

    # Plot the approximation results
    plt.plot(s_values, results1, linestyle='-', marker='', label='First-Order Approximation')
    plt.plot(s_values, results2, linestyle='-', marker='', label='Second-Order Approximation')
    # plt.plot(s_values, results3, linestyle='-', marker='', label='Third-Order Approximation')
    plt.xlabel('s values')
    plt.ylabel('Approximation')
    plt.title('Asymptotic Approximations')
    plt.legend()
    plt.show()