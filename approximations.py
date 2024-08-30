from math import gamma
from numba import jit
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from sympy import exp, oo, Symbol, diff
from sympy.stats import (Poisson, Binomial, NegativeBinomial, LogNormal,
                         Weibull, Frechet, Pareto, E, cdf, sample, density)

def calculate_exp_frechet(severity_params):
    """
    Calculate the expected value of a Frechet distribution.

    Args:
        severity_params (tuple of float): Parameters for the Frechet distribution (alpha, beta, min_val).

    Returns:
        float: Expected value of the Frechet distribution.
    """
    alpha, beta, min_val = severity_params

    # Use scipy to evaluate expected value numerically
    def integrand(x):
        """
        Integrand function for numerical integration.

        Args:
            x (float): The value at which to evaluate the integrand.

        Returns:
            float: Value of the integrand at x.
        """
        return x * frechet_pdf(x, alpha, beta, min_val)

    exp_X, _ = quad(integrand, min_val, np.inf)

    return exp_X

def initialize_distributions(claims_distribution_name, claims_params, severity_distribution_name, severity_params):
    """
    Initialize the distributions based on names and parameters.

    Args:
        claims_distribution_name (str): Name of the claims distribution ('Poisson', 'Binomial', 'Negative Binomial').
        claims_params (tuple of float): Parameters for the claims distribution.
        severity_distribution_name (str): Name of the severity distribution ('Lognormal', 'Weibull', 'Frechet', 'Pareto').
        severity_params (tuple of float): Parameters for the severity distribution.

    Returns:
        tuple: Contains initialized distributions and parameters (N, X, exp_X, a, b).
    """

    # Define the a and b values from the a,b,0-class
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
    """
    Custom PDF for the Frechet distribution.

    Args:
        y (float): Value to compute the PDF for.
        alpha (float): Shape parameter for the Frechet distribution.
        beta (float): Scale parameter for the Frechet distribution.
        min_val (float): Minimum value for the Frechet distribution.

    Returns:
        float: PDF value at y.
    """
    return (alpha / beta) * ((y - min_val) / beta) ** (-1 - alpha) * np.exp(-((y - min_val) / beta) ** (-alpha))

@jit
def frechet_cdf(y, alpha, beta, min_val):
    """
    Custom CDF for the Frechet distribution.

    Args:
        y (float): Value to compute the CDF for.
        alpha (float): Shape parameter for the Frechet distribution.
        beta (float): Scale parameter for the Frechet distribution.
        min_val (float): Minimum value for the Frechet distribution.

    Returns:
        float: CDF value at y.
    """
    return np.exp(-((y - min_val) / beta) ** (-alpha))

@jit
def weibull_pdf(s, lambda_, k):
    """
    Custom PDF for the Weibull distribution.

    Args:
        s (float): Value to compute the PDF for.
        lambda_ (float): Scale parameter for the Weibull distribution.
        k (float): Shape parameter for the Weibull distribution.

    Returns:
        float: PDF value at s.
    """
    return (k / lambda_) * (s / lambda_) ** (k - 1) * np.exp(-(s / lambda_) ** k)

@jit
def weibull_cdf(s, lambda_, k):
    """
    Custom CDF calculation for the Weibull distribution.

    Args:
        s (float): Value to compute the CDF for.
        lambda_ (float): Scale parameter for the Weibull distribution.
        k (float): Shape parameter for the Weibull distribution.

    Returns:
        float: CDF value at s.
    """
    return 1 - np.exp(-(s / lambda_) ** k)

@jit
def pareto_pdf(s, alpha, xm):
    """
    Custom PDF for the Pareto distribution.

    Args:
        s (float): Value to compute the PDF for.
        alpha (float): Shape parameter for the Pareto distribution.
        xm (float): Scale parameter for the Pareto distribution.

    Returns:
        float: PDF value at s.
    """
    return (alpha * xm ** alpha) / s ** (alpha + 1)
@jit
def pareto_cdf(s, alpha, xm):
    """
    Custom CDF calculation for the Pareto distribution.

    Args:
        s (float): Value to compute the CDF for.
        alpha (float): Shape parameter for the Pareto distribution.
        xm (float): Scale parameter for the Pareto distribution.

    Returns:
        float: CDF value at s.
    """
    return 1 - (xm / s) ** alpha

@jit
def lognormal_pdf(s, mu, sigma):
    """
    Custom PDF for the Lognormal distribution.

    Args:
        s (float): Value to compute the PDF for.
        mu (float): Mean of the underlying normal distribution.
        sigma (float): Standard deviation of the underlying normal distribution.

    Returns:
        float: PDF value at s.
    """
    return (1 / (s * sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((np.log(s) - mu) / sigma) ** 2)

# No @jit decorator here, because norm is not supported by Numba
def lognormal_cdf(s, mu, sigma):
    """
    Custom CDF for the Lognormal distribution.

    Args:
        s (float): Value to compute the CDF for.
        mu (float): Mean of the underlying normal distribution.
        sigma (float): Standard deviation of the underlying normal distribution.

    Returns:
        float: CDF value at s.
    """
    return norm.cdf((np.log(s) - mu) / sigma)

"""
def panjer_recursion_bounds(a, b, s_values, claim_amount, claim_distribution_name, claim_params,
                            severity_distribution_name, severity_params):
    '''
    Generalized Panjer recursion for upper and lower bounds using dictionaries.
    Computes both PDF and CDF values.

    :param a: Panjer distribution parameter a
    :param b: Panjer distribution parameter b
    :param s_values: Array or list of s_values to compute
    :param claim_amount: Claim amount for density calculation
    :param claim_distribution_name: Distribution name for claims
    :param claim_params: Parameters for the claim distribution
    :param severity_distribution_name: Severity distribution name
    :param severity_params: Parameters for the severity distribution
    :return: Dictionary of s_values with corresponding PDF and CDF values
    '''

    s_values = np.array(s_values)
    step_size = s_values[1] - s_values[0]

    # Precompute the f values based on severity_distribution_name
    def compute_f_values(s_values, step_size, severity_distribution_name, severity_params):
        if severity_distribution_name == 'Weibull':
            lambda_, k = severity_params
            f_values = weibull_cdf(s_values + step_size, lambda_, k) - weibull_cdf(s_values, lambda_, k)

        elif severity_distribution_name == 'Frechet':
            alpha, beta, min_val = severity_params
            f_values = frechet_cdf(s_values + step_size, alpha, beta, min_val) - frechet_pdf(s_values, alpha, beta,
                                                                                             min_val)

        elif severity_distribution_name == 'Pareto':
            alpha, xm = severity_params
            f_values = pareto_cdf(s_values + step_size, alpha, xm) - pareto_cdf(s_values, alpha, xm)

        elif severity_distribution_name == 'Lognormal':
            mu, sigma = severity_params
            f_values = lognormal_cdf(s_values + step_size, mu, sigma) - lognormal_cdf(s_values, mu, sigma)

        else:
            raise ValueError("Invalid severity distribution name")

        return f_values

    f_values = compute_f_values(s_values, step_size, severity_distribution_name, severity_params)

    # Initialize g values (PDF) and CDF
    g = np.zeros_like(s_values)
    g[0] = density(claim_amount)(0).evalf()  # Initial probability value for P(S=0)

    # Vectorized Panjer recursion
    for k in range(1, len(s_values)):
        s = s_values[k]
        k_index = np.searchsorted(s_values, s, side='left')
        if k_index < len(s_values):
            term_contributions = np.zeros(k_index)
            for j in range(1, k_index + 1):
                key = s - j * step_size
                key_index = np.searchsorted(s_values, key, side='left')
                if key_index < len(s_values) and np.isclose(s_values[key_index], key):
                    term_contributions[j - 1] = (a + b * j / k) * g[key_index] * f_values[
                        np.searchsorted(s_values, j * step_size, side='left')]

            g[k_index] = np.sum(term_contributions)

    # Compute CDF values by cumulative sum of PDF values
    cdf = np.cumsum(g)

    # Print debug information
    print("PDF Values:")
    for i, val in enumerate(g):
        print(f"s={s_values[i]:.3f}, PDF={val:.5f}, CDF={cdf[i]:.5f}")

    # Convert g and cdf to dictionary format
    result_dict = {
        s: {'pdf': g[i], 'cdf': cdf[i]} for i, s in enumerate(s_values)
    }

    return result_dict"""

def first_Order_Approximation(amount, severity, severity_distribution_name, severity_params, s_values):
    """
    Calculate the first-order approximation for the given parameters.

    Args:
        amount (Distribution): The claim amount distribution.
        severity (Distribution): The severity distribution.
        severity_distribution_name (str): The name of the severity distribution ('Weibull', 'Frechet', 'Pareto', 'Lognormal').
        severity_params (tuple): Parameters for the severity distribution.
        s_values (np.ndarray or list): The s-values for which to compute the approximation.

    Returns:
        np.ndarray: The first-order approximation results.
    """
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
    """
    Calculate the second-order approximation for the given parameters.

    Args:
        amount (Distribution): The claim amount distribution.
        severity (Distribution): The severity distribution.
        severity_distribution_name (str): The name of the severity distribution ('Frechet', 'Pareto', 'Lognormal', 'Weibull').
        severity_params (tuple): Parameters for the severity distribution.
        s_values (np.ndarray or list): The s-values for which to compute the approximation.
        results1 (np.ndarray): Results from the first-order approximation.
        exp_X (float): Expected value of the severity distribution.

    Returns:
        np.ndarray: The second-order approximation results.
    """
    exp_comb = E(amount * (amount - 1)).evalf()

    if severity_distribution_name == "Frechet":
        alpha, beta, min_val = severity_params
        eval_s = frechet_pdf(s_values, alpha, beta, min_val)
        integr_s = np.array([quad(lambda y: 1 - frechet_cdf(y, alpha, beta, min_val), s_values[0], s)[0] for s in s_values])
        if alpha == 1:
            results2 = 2 * exp_comb * eval_s * integr_s
        elif 0 < alpha < 1:
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
        eval_s = lognormal_pdf(s_values, mu, sigma)
        results2 = 2 * exp_comb * exp_X * eval_s
    else:  # Weibull Case
        lambda_, k = severity_params
        eval_s = weibull_pdf(s_values, lambda_, k)
        results2 = 2 * exp_comb * exp_X * eval_s

    return results1 + results2

def numerical_derivative(f, x, *params, epsilon=1e-5):
    """
    Compute the numerical derivative of a function using finite differences.

    Args:
        f (callable): The function for which to compute the derivative.
        x (float): The point at which to compute the derivative.
        *params: Parameters to pass to the function f.
        epsilon (float): The finite difference step size.

    Returns:
        float: The numerical derivative at x.
    """
    return (f(x + epsilon, *params) - f(x - epsilon, *params)) / (2 * epsilon)
def third_Order_Approximation(amount, severity, severity_distribution_name, severity_params, s_values, results1, exp_X):
    """
    Calculate the third-order approximation for the given parameters.

    Args:
        amount (Distribution): The claim amount distribution.
        severity (Distribution): The severity distribution.
        severity_distribution_name (str): The name of the severity distribution ('Weibull', 'Frechet', 'Pareto', 'Lognormal').
        severity_params (tuple): Parameters for the severity distribution.
        s_values (np.ndarray or list): The s-values for which to compute the approximation.
        results1 (np.ndarray): Results from the first-order approximation.
        exp_X (float): Expected value of the severity distribution.

    Returns:
        np.ndarray: The third-order approximation results.
    """
    j = 1
    N_samples = 100  # Number of samples for Monte Carlo simulation

    # Monte Carlo simulation
    expected_value_samples = []
    try:
        for _ in range(N_samples):
            n = sample(amount)
            if n > 1:
                X_values = [severity for _ in range(n - 1)]
                sum_X = sum(X_values)
                eval_sum_X = E(sum_X ** (j + 1)).evalf()

                max_value = 1e+300  # Define a threshold for maximum value
                if eval_sum_X > max_value:
                    print(f"Warning: eval_sum_X is too large: {eval_sum_X}")
                    eval_sum_X = max_value / 1000 # Cap the value to prevent overflow

                testResult = n * eval_sum_X
                if testResult > max_value:
                    print(f"Warning: Result is too large: {testResult}")
                    testResult = max_value  # Cap the result if it is too large

            expected_value_samples.append(testResult)

        # Calculate the expected value
        expected_value = np.mean(expected_value_samples)

        if severity_distribution_name == 'Weibull':
            diffX = np.array([numerical_derivative(weibull_pdf, s, *severity_params) for s in s_values])
        elif severity_distribution_name == 'Frechet':
            diffX = np.array([numerical_derivative(frechet_pdf, s, *severity_params) for s in s_values])
        elif severity_distribution_name == 'Pareto':
            diffX = np.array([numerical_derivative(pareto_pdf, s, *severity_params) for s in s_values])
        elif severity_distribution_name == 'Lognormal':
            diffX = np.array([numerical_derivative(lognormal_pdf, s, *severity_params) for s in s_values])
        else:
            raise ValueError("Invalid severity distribution name")

    except:
        print("Value error, higher moments don't exist for the chosen parameters")
        return 0

    return (diffX * expected_value * 1/2)

def numerical_second_derivative(f, x, *params, epsilon=1e-5):
    """
    Compute the second derivative of a function using finite differences.

    Args:
        f (callable): The function for which to compute the second derivative.
        x (float): The point at which to compute the derivative.
        *params: Parameters to pass to the function f.
        epsilon (float): The finite difference step size.

    Returns:
        float: The second derivative at x.
    """
    return (numerical_derivative(f, x + epsilon, *params) - numerical_derivative(f, x - epsilon, *params)) / (2 * epsilon)

def fourth_Order_Approximation(amount, severity, severity_distribution_name, severity_params, s_values, results1, exp_X):
    """
    Calculate the fourth-order approximation for the given parameters.

    Args:
        amount (Distribution): The claim amount distribution.
        severity (Distribution): The severity distribution.
        severity_distribution_name (str): The name of the severity distribution ('Weibull', 'Frechet', 'Pareto', 'Lognormal').
        severity_params (tuple): Parameters for the severity distribution.
        s_values (np.ndarray or list): The s-values for which to compute the approximation.
        results1 (np.ndarray): Results from the first-order approximation.
        exp_X (float): Expected value of the severity distribution.

    Returns:
        np.ndarray: The fourth-order approximation results.
    """
    j = 2
    N_samples = 100  # Number of samples for Monte Carlo simulation

    # Monte Carlo simulation
    expected_value_samples = []
    try:
        for _ in range(N_samples):
            n = sample(amount)
            if n > 1:
                X_values = [severity for _ in range(n - 1)]
                sum_X = sum(X_values)
                eval_sum_X = E(sum_X ** (j + 1)).evalf()
                expected_value_samples.append(n * eval_sum_X)

        # Calculate the expected value
        expected_value = np.mean(expected_value_samples)

        if severity_distribution_name == 'Weibull':
            diffX = np.array([numerical_second_derivative(weibull_pdf, s, *severity_params) for s in s_values])
        elif severity_distribution_name == 'Frechet':
            diffX = np.array([numerical_second_derivative(frechet_pdf, s, *severity_params) for s in s_values])
        elif severity_distribution_name == 'Pareto':
            diffX = np.array([numerical_second_derivative(pareto_pdf, s, *severity_params) for s in s_values])
        elif severity_distribution_name == 'Lognormal':
            diffX = np.array([numerical_second_derivative(lognormal_pdf, s, *severity_params) for s in s_values])
        else:
            raise ValueError("Invalid severity distribution name")

    except:
        print("Value error, higher moments don't exist for the chosen parameters")
        return 0

    return (diffX * expected_value)


def numerical_third_derivative(f, x, *params, epsilon=1e-2):
    """
    Compute the third derivative of a function using finite differences.

    Args:
        f (callable): The function for which to compute the third derivative.
        x (float): The point at which to compute the derivative.
        *params: Parameters to pass to the function f.
        epsilon (float): The finite difference step size.

    Returns:
        float: The third derivative at x.
    """
    return (numerical_second_derivative(f, x + epsilon, *params) - numerical_second_derivative(f, x - epsilon, *params)) / (2 * epsilon)

def fifth_Order_Approximation(amount, severity, severity_distribution_name, severity_params, s_values, results1, exp_X):
    """
    Calculate the fifth-order approximation for the given parameters.

    Args:
        amount (Distribution): The claim amount distribution.
        severity (Distribution): The severity distribution.
        severity_distribution_name (str): The name of the severity distribution ('Weibull', 'Frechet', 'Pareto', 'Lognormal').
        severity_params (tuple): Parameters for the severity distribution.
        s_values (np.ndarray or list): The s-values for which to compute the approximation.
        results1 (np.ndarray): Results from the first-order approximation.
        exp_X (float): Expected value of the severity distribution.

    Returns:
        np.ndarray: The fifth-order approximation results.
    """
    j = 3
    N_samples = 100  # Number of samples for Monte Carlo simulation

    # Monte Carlo simulation
    expected_value_samples = []
    try:
        for _ in range(N_samples):
            n = sample(amount)
            if n > 1:
                X_values = [severity for _ in range(n - 1)]
                sum_X = sum(X_values)
                eval_sum_X = E(sum_X ** (j + 1)).evalf()
                expected_value_samples.append(n * eval_sum_X)

        # Calculate the expected value
        expected_value = np.mean(expected_value_samples)

        if severity_distribution_name == 'Weibull':
            diffX = np.array([numerical_third_derivative(weibull_pdf, s, *severity_params) for s in s_values])
        elif severity_distribution_name == 'Frechet':
            diffX = np.array([numerical_third_derivative(frechet_pdf, s, *severity_params) for s in s_values])
        elif severity_distribution_name == 'Pareto':
            diffX = np.array([numerical_third_derivative(pareto_pdf, s, *severity_params) for s in s_values])
        elif severity_distribution_name == 'Lognormal':
            diffX = np.array([numerical_third_derivative(lognormal_pdf, s, *severity_params) for s in s_values])
        else:
            raise ValueError("Invalid severity distribution name")

    except:
        print("Value error, higher moments don't exist for the chosen parameters")
        return 0

    return (diffX * expected_value)

def numerical_fourth_derivative(f, x, *params, epsilon=1e-2):
    """
    Compute the fourth derivative of a function using finite differences.

    Args:
        f (callable): The function for which to compute the fourth derivative.
        x (float): The point at which to compute the derivative.
        *params: Parameters to pass to the function f.
        epsilon (float): The finite difference step size.

    Returns:
        float: The fourth derivative at x.
    """
    return (numerical_third_derivative(f, x + epsilon, *params) - numerical_third_derivative(f, x - epsilon, *params)) / (2 * epsilon)

def sixth_Order_Approximation(amount, severity, severity_distribution_name, severity_params, s_values, results1, exp_X):
    """
    Calculate the sixth-order approximation for the given parameters.

    Args:
        amount (Distribution): The claim amount distribution.
        severity (Distribution): The severity distribution.
        severity_distribution_name (str): The name of the severity distribution ('Weibull', 'Frechet', 'Pareto', 'Lognormal').
        severity_params (tuple): Parameters for the severity distribution.
        s_values (np.ndarray or list): The s-values for which to compute the approximation.
        results1 (np.ndarray): Results from the first-order approximation.
        exp_X (float): Expected value of the severity distribution.

    Returns:
        np.ndarray: The sixth-order approximation results.
    """
    j = 4
    N_samples = 100  # Number of samples for Monte Carlo simulation

    # Monte Carlo simulation
    expected_value_samples = []
    try:
        for _ in range(N_samples):
            n = sample(amount)
            if n > 1:
                X_values = [severity for _ in range(n - 1)]
                sum_X = sum(X_values)
                eval_sum_X = E(sum_X ** (j + 1)).evalf()
                expected_value_samples.append(n * eval_sum_X)

        # Calculate the expected value
        expected_value = np.mean(expected_value_samples)

        if severity_distribution_name == 'Weibull':
            diffX = np.array([numerical_fourth_derivative(weibull_pdf, s, *severity_params) for s in s_values])
        elif severity_distribution_name == 'Frechet':
            diffX = np.array([numerical_fourth_derivative(frechet_pdf, s, *severity_params) for s in s_values])
        elif severity_distribution_name == 'Pareto':
            diffX = np.array([numerical_fourth_derivative(pareto_pdf, s, *severity_params) for s in s_values])
        elif severity_distribution_name == 'Lognormal':
            diffX = np.array([numerical_fourth_derivative(lognormal_pdf, s, *severity_params) for s in s_values])
        else:
            raise ValueError("Invalid severity distribution name")

    except:
        print("Value error, higher moments don't exist for the chosen parameters")
        return 0

    return (diffX * expected_value)


def doCalculations(claims_distribution_name, claims_params, severity_distribution_name, severity_params):
    """
    Perform calculations for different orders of approximation based on the given parameters.

    Args:
        claims_distribution_name (str): The name of the claims distribution.
        claims_params (tuple): Parameters for the claims distribution.
        severity_distribution_name (str): The name of the severity distribution ('Weibull', 'Frechet', 'Pareto', 'Lognormal').
        severity_params (tuple): Parameters for the severity distribution.

    Returns:
        tuple: (s_values, results1, results2, final_results3, final_results4, final_results5, final_results6)
               Where:
               - s_values: The range of s-values used for the calculations.
               - results1: The first-order approximation results.
               - results2: The second-order approximation results.
               - final_results3: The third-order approximation results (or None if not computed).
               - final_results4: The fourth-order approximation results (or None if not computed).
               - final_results5: The fifth-order approximation results (or None if not computed).
               - final_results6: The sixth-order approximation results (or None if not computed).
    """
    # Define range of s values
    # Add Epsilon to prevent division by zero in the exponent
    epsilon = 1e-10
    if severity_distribution_name == "Pareto":
        s_min = severity_params[1] + epsilon
    elif severity_distribution_name == "Frechet":
        s_min = severity_params[2] + epsilon
    else:
        s_min = 1 + epsilon
    s_max = 100001
    num_points = 100001  # Number of points to plot
    s_values = np.linspace(s_min, s_max, num_points)

    # Initialize distributions
    N, X, exp_X, a, b = initialize_distributions(claims_distribution_name, claims_params, severity_distribution_name,
                                    severity_params)

    # Evaluate first-order approximation
    results1 = first_Order_Approximation(N, X, severity_distribution_name, severity_params, s_values)

    # Evaluate second- and higher-order approximations
    results2 = second_Order_Approximation(N, X, severity_distribution_name, severity_params, s_values, results1, exp_X)

    # Check if the higher order approximations return a result
    final_results3 = final_results4 = final_results5 = final_results6 = None # set the approximation value to None to handle in interface

    # Don't do higher order approximations for Frechet, it takes too long because of symbolic evaluation of sum of random variables
    if severity_distribution_name != "Frechet":
        results3 = third_Order_Approximation(N, X, severity_distribution_name, severity_params, s_values, results1, exp_X)

        if isinstance(results3, np.ndarray):
            final_results3 = results2 - results3
            results4 = fourth_Order_Approximation(N, X, severity_distribution_name, severity_params, s_values, results1,
                                                  exp_X)
            print("3 done")
            if isinstance(results4, np.ndarray):
                final_results4 = results3 + results4
                results5 = fifth_Order_Approximation(N, X, severity_distribution_name, severity_params, s_values, results1,
                                                     exp_X)
                print("4 done")
                if isinstance(results5, np.ndarray):
                    final_results5 = results4 - results5
                    results6 = sixth_Order_Approximation(N, X, severity_distribution_name, severity_params, s_values,
                                                         results1, exp_X)
                    print("5 done")
                    if isinstance(results6, np.ndarray):
                        final_results6 = results5 + results6
                        print("6 done")

    return s_values, results1, results2,final_results3, final_results4, final_results5, final_results6

