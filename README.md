# Higher-Order Approximations for Compound Sum Distributions

This project provides a tool for performing first- to higher-order approximations of various probability distributions. It includes a computational module (`approximations.py`) for the approximations and an interactive graphical interface (`interface.py`) for selecting distributions, setting parameters, and plotting the results.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Features](#features)
- [License](#license)
- [Author](#author)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/evanhecke/StatsOrderExpansions
   cd StatsOrderExpansions
   ```

2. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the interface script to interact with the tool:
   ```bash
   python interface.py
   ```
   
2. Follow the on-screen instructions to:
   - Choose a distribution (e.g., Poisson, Binomial, Weibull).
   - Set the necessary parameters (e.g., lambda, trials, probability).
   - Plot the approximations based on your selected settings.

## Project Structure

- **`approximations.py`**: This module contains all the computations for first- to higher-order approximations of selected probability distributions. It includes handling various approximation methods and checks for oscillatory or missing approximations.
  
- **`interface.py`**: This script provides an interactive GUI interface, allowing users to select distributions, input parameters, and visualize computed approximations. The tool supports multiple distributions and includes input validation and error handling.

## Features

- Compute first- to higher-order approximations for various probability distributions.
- Interactive GUI for easy configuration of distributions and parameters.
- Plotting capabilities to visualize approximations, with warnings for missing or oscillatory results.

## License

This project does not currently have a license. Please contact the author if you have questions about usage.

## Author

- **Evert Van Hecke** - [evanhecke](https://github.com/evanhecke)

