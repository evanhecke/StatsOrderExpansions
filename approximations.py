from math import gamma
from numba import jit
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from sympy import exp, oo
from sympy.stats import (Poisson, Binomial, NegativeBinomial, LogNormal,
                         Weibull, Frechet, Pareto, E, cdf)

def calculate_exp_frechet(severity_params):
    alpha, beta, min_val = severity_params

    # Use scipy to evaluate expected value numerically
    def integrand(x):
        return x * frechet_pdf(x, alpha, beta, min_val)

    exp_X, _ = quad(integrand, min_val, np.inf)

    return exp_X

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
        exp_X = calculate_exp_frechet(severity_params)
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

@jit
def frechet_pdf(y, alpha, beta, min_val):
    """Custom PDF for the Frechet distribution."""
    return (alpha / beta) * ((y - min_val) / beta) ** (-1 - alpha) * np.exp(-((y - min_val) / beta) ** (-alpha))

@jit
def frechet_cdf(y, alpha, beta, min_val):
    """Custom CDF for the Frechet distribution."""
    return np.exp(-((y - min_val) / beta) ** (-alpha))

@jit
def weibull_pdf(s, lambda_, k):
    """Custom PDF for the Weibull distribution."""
    return (k / lambda_) * (s / lambda_) ** (k - 1) * np.exp(-(s / lambda_) ** k)

@jit
def weibull_cdf(s, lambda_, k):
    """Custom CDF calculation for the Weibull distribution."""
    return 1 - np.exp(-(s / lambda_) ** k)

@jit
def pareto_pdf(s, alpha, xm):
    """Custom PDF for the Pareto distribution."""
    return (alpha * xm ** alpha) / s ** (alpha + 1)
@jit
def pareto_cdf(s, alpha, xm):
    """Custom CDF calculation for the Pareto distribution."""
    return 1 - (xm / s) ** alpha

@jit
def lognormal_pdf(s, mu, sigma):
    """Custom PDF for the Lognormal distribution."""
    return (1 / (s * sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((np.log(s) - mu) / sigma) ** 2)

@jit
def lognormal_cdf(s, mu, sigma):
    """Custom CDF for the Lognormal distribution."""
    return norm.cdf((np.log(s) - mu) / sigma)


def first_Order_Approximation(amount, severity, severity_distribution_name, severity_params, s_values):
    if severity_distribution_name == 'Weibull':
        lambda_, k = severity_params
        tailX = 1 - weibull_cdf(s_values, lambda_, k)
    elif severity_distribution_name == 'Frechet':
        alpha, beta, min_val = severity_params
        tailX = 1 - frechet_cdf(s_values, alpha, beta, min_val)
    elif severity_distribution_name == 'Pareto':
        alpha, xm = severity_params
        tailX = 1 - pareto_cdf(s_values, alpha, xm)
    elif severity_distribution_name == 'Lognormal':
        mu, sigma = severity_params
        tailX = 1 - lognormal_cdf(s_values, mu, sigma)
    else:
        raise ValueError("Invalid severity distribution name")

    results1 = E(amount).evalf() * tailX
    results1 = np.where(np.iscomplex(results1), results1.real, results1)

    return results1


def second_Order_Approximation(amount, severity, severity_distribution_name, severity_params, s_values, results1, exp_X):
    exp_comb = E(amount * (amount - 1)).evalf()

    if severity_distribution_name == "Frechet":
        alpha, beta, min_val = severity_params
        eval_s = frechet_pdf(s_values, alpha, beta, min_val)
        if alpha == 1:
            integr_s = np.array([quad(lambda y: 1 - frechet_cdf(y, alpha, beta, min_val), s_values[0], s)[0] for s in s_values])
            results2 = 2 * exp_comb * eval_s * integr_s
        elif 0 < alpha < 1:
            integr_s = np.array([quad(lambda y: 1 - frechet_cdf(y, alpha, beta, min_val), s_values[0], s)[0] for s in s_values])
            expr = - (2 - alpha) * gamma(2 - alpha) / ((alpha - 1) * gamma(3 - 2 * alpha))
            results2 = exp_comb * expr * eval_s * integr_s
        else:
            results2 = 2 * exp_comb * exp_X * eval_s
    elif severity_distribution_name == "Pareto":
        alpha, xm = severity_params
        integr_s = np.array([quad(lambda y: 1 - pareto_cdf(y, alpha, xm), s_values[0], s)[0] for s in s_values])
        eval_s = pareto_pdf(s_values, alpha, xm)
        if alpha == 1:
            results2 = 2 * exp_comb * eval_s * integr_s
        elif 0 < alpha < 1:
            expr = - (2 - alpha) * gamma(2 - alpha) / ((alpha - 1) * gamma(3 - 2 * alpha))
            results2 = exp_comb * expr * eval_s * integr_s
        else:
            results2 = 2 * exp_comb * exp_X * eval_s
    elif severity_distribution_name == "Lognormal":
        mu, sigma = severity_params
        integr_s = np.array([quad(lambda y: 1 - lognormal_cdf(y, mu, sigma), s_values[0], s)[0] for s in s_values])
        eval_s = lognormal_pdf(s_values, mu, sigma)
        results2 = 2 * exp_comb * exp_X * eval_s
    else:  # Weibull Case
        lambda_, k = severity_params
        eval_s = weibull_pdf(s_values, lambda_, k)
        results2 = 2 * exp_comb * exp_X * eval_s

    return results1 + results2


# def higher_Order_Approximations(amount, severity, severity_distribution_name, severity_params, s_values, results1, exp_X):



def doCalculations(claims_distribution_name, claims_params, severity_distribution_name, severity_params):

    # print(Expectation(N).doit())
    # print((P(N > 4)).evalf(3))
    # Z = Normal('Z', 5, 2)
    # print(cdf(N)(1).evalf())
    # print(cdf(Z)(5).evalf())

    # Define range of s values
    # Add Epsilon to prevent division by zero in the exponent
    epsilon = 1e-10
    if severity_distribution_name == "Pareto":
        s_min = severity_params[1] + epsilon
    elif severity_distribution_name == "Frechet":
        s_min = severity_params[2] + epsilon
    else:
        s_min = 1 + epsilon
    s_max = 1001
    num_points = 101  # Number of points to plot
    s_values = np.linspace(s_min, s_max, num_points)

    # Initialize distributions
    N, X, exp_X, a, b = initialize_distributions(claims_distribution_name, claims_params, severity_distribution_name,
                                    severity_params)
    print("Claim amount distribution:", cdf(N))
    print("Exp_N:", E(N).evalf())
    #print("Severity distribution:", cdf(X))

    # Evaluate first-order approximation
    results1 = first_Order_Approximation(N, X, severity_distribution_name, severity_params, s_values)

    # Evaluate second-order approximation
    results2 = second_Order_Approximation(N, X, severity_distribution_name, severity_params, s_values, results1, exp_X)

    # results3 = third_Order_Approximation(N, X, s_values, results2)

    # Debug for complex values
    #print("Results1:", results1)
    #print("Exp_X:", exp_X)
    #print("Results2:", results2)

    # Plot the approximation results
    plt.plot(s_values, results1, linestyle='-', marker='', label='First-Order Approximation')
    plt.plot(s_values, results2, linestyle='-', marker='', label='Second-Order Approximation')
    # plt.plot(s_values, results3, linestyle='-', marker='', label='Third-Order Approximation')
    plt.xlabel('s values')
    plt.ylabel('Approximation')
    plt.title('Asymptotic Approximations')
    plt.legend()
    plt.show()