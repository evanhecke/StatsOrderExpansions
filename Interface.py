import tkinter as tk
import matplotlib.pyplot as plt
from tkinter import ttk
from approximations import doCalculations
from PIL import Image, ImageTk
import io


class DistributionCalculator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Distribution Calculator")

        self.claims_distribution_var = tk.StringVar()
        self.severity_distribution_var = tk.StringVar()

        self.claims_parameters = {}
        self.severity_parameters = {}

        self.claims_distributions = {
            "Poisson": "poisson",
            "Binomial": "binom",
            "Negative Binomial": "nbinom"
        }  # Allowed claims distributions

        self.severity_distributions = {
            "Lognormal": "lognorm",
            "Weibull": "weibull_min",
            "Pareto": "pareto",
            "Frechet": "genextreme",
            # "Exponential": "expon",
            # "Gamma": "gamma",
            # "Normal": "norm"
        }  # Allowed severity distributions

        # PDF formulas in LaTeX format
        self.pdf_formulas = {
            "Poisson": r"$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k=0,1,2,\ldots$",
            "Binomial": r"$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k=0,1,\ldots,n$",
            "Negative Binomial": r"$P(X = k) = \binom{k+r-1}{k} p^r (1-p)^{k}, \quad k=0,1,\ldots$",
            "Lognormal": r"$f(x) = \frac{1}{x \sigma \sqrt{2\pi}} e^{-\frac{(\ln x - \mu)^2}{2\sigma^2}}, \quad x > 0$",
            "Weibull": r"$f(x) = \frac{k}{\lambda} \left(\frac{x}{\lambda}\right)^{k-1} e^{-\left(\frac{x}{\lambda}\right)^k}, \quad x \geq 0$",
            "Pareto": r"$f(x) = \frac{\alpha x_m^\alpha}{x^{\alpha + 1}}, \quad x \geq x_m$",
            "Frechet": r"$f(x) = \frac{\alpha}{\beta} \left( \frac{x-x_m}{\beta} \right)^{-\alpha-1} e^{-\left( \frac{x-x_m}{\beta} \right)^{-\alpha}}, \quad x > 0$",
            # "Exponential": r"$f(x) = \lambda e^{-\lambda x}, \quad x \geq 0$",
            # "Gamma": r"$f(x) = \frac{x^{k-1} e^{-\frac{x}{\theta}}}{\theta^k \Gamma(k)}, \quad x \geq 0$",
            # "Normal": r"$f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}$"
        }

        self.setup_ui()

    def setup_ui(self):
        # Left side - Claims Distribution
        claims_frame = ttk.Frame(self)
        claims_frame.grid(row=0, column=0, padx=10, pady=5)

        claims_distribution_label = ttk.Label(claims_frame, text="Select Claims Distribution:")
        claims_distribution_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        self.claims_distribution_combo = ttk.Combobox(
            claims_frame,
            values=list(self.claims_distributions.keys()),
            textvariable=self.claims_distribution_var
        )
        self.claims_distribution_combo.bind("<<ComboboxSelected>>", self.generate_claims_parameter_inputs)
        self.claims_distribution_combo.grid(row=1, column=0, padx=10, pady=5)

        # Label to display the claims PDF formula
        self.claims_pdf_formula_label = ttk.Label(claims_frame)
        self.claims_pdf_formula_label.grid(row=1, column=1, padx=10, pady=5)

        # Right side - Severity Distribution
        severity_frame = ttk.Frame(self)
        severity_frame.grid(row=0, column=1, padx=10, pady=5)

        severity_distribution_label = ttk.Label(severity_frame, text="Select Severity Distribution:")
        severity_distribution_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        self.severity_distribution_combo = ttk.Combobox(
            severity_frame,
            values=list(self.severity_distributions.keys()),
            textvariable=self.severity_distribution_var
        )
        self.severity_distribution_combo.bind("<<ComboboxSelected>>", self.generate_severity_parameter_inputs)
        self.severity_distribution_combo.grid(row=1, column=0, padx=10, pady=5)

        # Label to display the severity PDF formula
        self.severity_pdf_formula_label = ttk.Label(severity_frame)
        self.severity_pdf_formula_label.grid(row=1, column=1, padx=10, pady=5)

        # Button for calculations
        self.calculate_button = ttk.Button(self, text="Calculate", command=self.calculate)
        self.calculate_button.grid(row=2, column=0, columnspan=2, padx=10, pady=5)

    def generate_claims_parameter_inputs(self, event=None):
        # Remove previous parameter frame
        if hasattr(self, "claims_parameter_frame"):
            self.claims_parameter_frame.destroy()

        claims_distribution_name = self.claims_distribution_var.get()

        # Create a new frame for parameters
        self.claims_parameter_frame = ttk.Frame(self)
        self.claims_parameter_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        self.claims_parameters = {}

        if claims_distribution_name == "Poisson":
            param_label = ttk.Label(self.claims_parameter_frame, text="λ (Lambda)")
            param_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
            param_entry = ttk.Entry(self.claims_parameter_frame)
            param_entry.grid(row=0, column=1, padx=10, pady=5)
            self.claims_parameters["λ (Lambda)"] = param_entry

        elif claims_distribution_name == "Binomial":
            param_label_n = ttk.Label(self.claims_parameter_frame, text="n (Trials)")
            param_label_n.grid(row=0, column=0, padx=10, pady=5, sticky="w")
            param_entry_n = ttk.Entry(self.claims_parameter_frame)
            param_entry_n.bind("<FocusOut>", self.validate_integer_positive)  # Validate n > 0
            param_entry_n.grid(row=0, column=1, padx=10, pady=5)
            self.claims_parameters["n (Trials)"] = param_entry_n

            param_label_p = ttk.Label(self.claims_parameter_frame, text="p (Probability)")
            param_label_p.grid(row=1, column=0, padx=10, pady=5, sticky="w")
            param_entry_p = ttk.Entry(self.claims_parameter_frame)
            param_entry_p.bind("<FocusOut>", self.validate_probability)  # Validate 0 ≤ p ≤ 1
            param_entry_p.grid(row=1, column=1, padx=10, pady=5)
            self.claims_parameters["p (Probability)"] = param_entry_p

        elif claims_distribution_name == "Negative Binomial":
            param_label_r = ttk.Label(self.claims_parameter_frame, text="r (Successes)")
            param_label_r.grid(row=0, column=0, padx=10, pady=5, sticky="w")
            param_entry_r = ttk.Entry(self.claims_parameter_frame)
            param_entry_r.bind("<FocusOut>", self.validate_integer_positive)  # Validate r > 0
            param_entry_r.grid(row=0, column=1, padx=10, pady=5)
            self.claims_parameters["r (Successes)"] = param_entry_r

            param_label_p = ttk.Label(self.claims_parameter_frame, text="p (Probability)")
            param_label_p.grid(row=1, column=0, padx=10, pady=5, sticky="w")
            param_entry_p = ttk.Entry(self.claims_parameter_frame)
            param_entry_p.bind("<FocusOut>", self.validate_probability)  # Validate 0 ≤ p ≤ 1
            param_entry_p.grid(row=1, column=1, padx=10, pady=5)
            self.claims_parameters["p (Probability)"] = param_entry_p

        self.update_claims_pdf_formula(claims_distribution_name)

    def generate_severity_parameter_inputs(self, event=None):
        # Remove previous parameter frame
        if hasattr(self, "severity_parameter_frame"):
            self.severity_parameter_frame.destroy()

        severity_distribution_name = self.severity_distribution_var.get()

        # Create a new frame for parameters
        self.severity_parameter_frame = ttk.Frame(self)
        self.severity_parameter_frame.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        self.severity_parameters = {}

        if severity_distribution_name == "Lognormal":
            param_names = ["μ (Mu)", "σ (Sigma)"]
        elif severity_distribution_name == "Weibull":
            param_names = ["λ (Lambda)", "k (Shape)"]
        elif severity_distribution_name == "Pareto":
            param_names = ["α (Alpha)", "x_m (Minimum)"]
        elif severity_distribution_name == "Frechet":
            param_names = ["α (Alpha)", "β (Beta)", "x_m (Minimum)"]
        else:
            return

        for i, param_name in enumerate(param_names):
            param_label = ttk.Label(self.severity_parameter_frame, text=param_name)
            param_label.grid(row=i, column=0, padx=10, pady=5, sticky="w")

            param_entry = ttk.Entry(self.severity_parameter_frame)

            if severity_distribution_name == "Weibull":
                if param_name == "k (Shape)":
                    param_entry.bind("<FocusOut>", self.validate_weibull_k)
                elif param_name == "λ (Lambda)":
                    param_entry.bind("<FocusOut>", lambda e: self.validate_strictly_positive(e, "λ (Lambda)"))
            elif severity_distribution_name == "Frechet":
                if param_name == "α (Alpha)":
                    param_entry.bind("<FocusOut>", lambda e: self.validate_strictly_positive(e, "α (Alpha)"))
                elif param_name == "β (Beta)":
                    param_entry.bind("<FocusOut>", lambda e: self.validate_strictly_positive(e, "β (Beta)"))
            elif severity_distribution_name == "Pareto":
                if param_name == "α (Alpha)":
                    param_entry.bind("<FocusOut>", lambda e: self.validate_strictly_positive(e, "α (Alpha)"))
                elif param_name == "x_m (Minimum)":
                    param_entry.bind("<FocusOut>", lambda e: self.validate_strictly_positive(e, "x_m (Minimum)"))
            elif severity_distribution_name == "Lognormal" and param_name == "σ (Sigma)":
                param_entry.bind("<FocusOut>", lambda e: self.validate_strictly_positive(e, "σ (Sigma)"))

            param_entry.grid(row=i, column=1, padx=10, pady=5)
            self.severity_parameters[param_name] = param_entry

        self.update_severity_pdf_formula(severity_distribution_name)

    def validate_weibull_k(self, event):
        """Validation function to ensure k < 1 for Weibull distribution."""
        entry = event.widget
        input_value = entry.get()
        try:
            value = float(input_value)
            if not (0 < value < 1):
                tk.messagebox.showerror(
                    "Invalid Input",
                    "For the Weibull distribution, k must be between 0 and 1.\n"
                    "This is necessary for the distribution to have a heavy tail."
                )
                entry.delete(0, tk.END)  # Clear invalid input
                return False
        except ValueError:
            tk.messagebox.showerror("Invalid Input", "Please enter a valid number for k.")
            entry.delete(0, tk.END)  # Clear invalid input
            return False

    def validate_strictly_positive(self, event, param_name=""):
        """Ensure the value is strictly positive."""
        entry = event.widget
        input_value = entry.get()

        if input_value.strip() == "":  # Skip empty inputs
            return True

        try:
            value = float(input_value)
            if value <= 0:
                tk.messagebox.showerror("Invalid Input", f"The value for {param_name} must be strictly positive.")
                entry.delete(0, tk.END)  # Clear invalid input
                return False
        except ValueError:
            tk.messagebox.showerror("Invalid Input", f"Please enter a valid positive number for {param_name}.")
            entry.delete(0, tk.END)  # Clear invalid input
            return False

    def validate_integer_positive(self, event):
        """Ensure the value is a positive integer."""
        entry = event.widget
        input_value = entry.get()

        if input_value.strip() == "":  # Skip empty inputs
            return True

        try:
            value = int(input_value)
            if value <= 0:
                tk.messagebox.showerror("Invalid Input", "The value must be a positive integer greater than zero.")
                entry.delete(0, tk.END)  # Clear invalid input
                return False
        except ValueError:
            tk.messagebox.showerror("Invalid Input", "Please enter a valid integer.")
            entry.delete(0, tk.END)  # Clear invalid input
            return False

    def validate_probability(self, event):
        """Ensure the value is a probability between 0 and 1 (inclusive)."""
        entry = event.widget
        input_value = entry.get()

        if input_value.strip() == "":  # Skip empty inputs
            return True

        try:
            value = float(input_value)
            if not (0 <= value <= 1):
                tk.messagebox.showerror("Invalid Input", "The probability must be between 0 and 1 (inclusive).")
                entry.delete(0, tk.END)  # Clear invalid input
                return False
        except ValueError:
            tk.messagebox.showerror("Invalid Input", "Please enter a valid number for the probability.")
            entry.delete(0, tk.END)  # Clear invalid input
            return False

    def update_calculate_button_state(self):
        """Enable or disable the calculate button based on input validity."""
        valid = True
        for param_entry in list(self.claims_parameters.values()) + list(self.severity_parameters.values()):
            if param_entry.get().strip() == "":
                valid = False
                break

        self.calculate_button.config(state=tk.NORMAL if valid else tk.DISABLED)

    def update_claims_pdf_formula(self, distribution_name):
        """Update the label with the claims PDF formula in LaTeX format."""
        formula = self.pdf_formulas.get(distribution_name, "")
        self.display_latex_formula(formula, self.claims_pdf_formula_label)

    def update_severity_pdf_formula(self, distribution_name):
        """Update the label with the severity PDF formula in LaTeX format."""
        formula = self.pdf_formulas.get(distribution_name, "")
        self.display_latex_formula(formula, self.severity_pdf_formula_label)

    def display_latex_formula(self, formula, label):
        """Render a LaTeX formula using matplotlib and display it in the given label."""
        fig, ax = plt.subplots(figsize=(4, 1))  # Adjust size as needed
        ax.text(0.5, 0.5, formula, size=16, ha='center', va='center')  # Reduced font size
        ax.axis('off')

        # Save to a BytesIO stream
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

        # Convert to an image and display it
        buf.seek(0)
        image = Image.open(buf)
        photo = ImageTk.PhotoImage(image)

        label.config(image=photo)
        label.image = photo  # Keep a reference to avoid garbage collection

    def calculate(self):
        claims_distribution_name = self.claims_distribution_var.get()
        severity_distribution_name = self.severity_distribution_var.get()

        # Check if both distributions are selected
        if not claims_distribution_name or not severity_distribution_name:
            tk.messagebox.showerror("Missing Distribution",
                                    "Please select both a claims distribution and a severity distribution.")
            return

        # Check if all claims parameters are filled
        for param_entry in self.claims_parameters.values():
            if not param_entry.get().strip():  # If any parameter is empty
                tk.messagebox.showerror("Missing Parameters",
                                        "Please fill in all parameters for the claims distribution.")
                return

        # Check if all severity parameters are filled
        for param_entry in self.severity_parameters.values():
            if not param_entry.get().strip():  # If any parameter is empty
                tk.messagebox.showerror("Missing Parameters",
                                        "Please fill in all parameters for the severity distribution.")
                return

        claims_params = [float(param_entry.get()) for param_entry in self.claims_parameters.values()]
        severity_params = [float(param_entry.get()) for param_entry in self.severity_parameters.values()]

        # Perform calculations with chosen distributions and parameters
        print("Claims Distribution:", claims_distribution_name)
        print("Claims Distribution Parameters:", claims_params)
        print("Severity Distribution:", severity_distribution_name)
        print("Severity Distribution Parameters:", severity_params)

        doCalculations(claims_distribution_name, claims_params, severity_distribution_name, severity_params)


if __name__ == "__main__":
    app = DistributionCalculator()
    app.mainloop()
