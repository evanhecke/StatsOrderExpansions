import tkinter as tk
from tkinter import ttk
from scipy import stats
from TailBehavior import doCalculations


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
            "Frechet": "genextreme"
        }  # Allowed severity distributions

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

        # Button for calculations
        self.calculate_button = ttk.Button(self, text="Calculate", command=self.calculate)
        self.calculate_button.grid(row=2, column=0, columnspan=2, padx=10, pady=5)

    def generate_claims_parameter_inputs(self, event=None):
        # Remove previous parameter frame
        if hasattr(self, "claims_parameter_frame"):
            self.claims_parameter_frame.destroy()

        claims_distribution_name = self.claims_distribution_var.get()
        claims_distribution = getattr(stats, self.claims_distributions[claims_distribution_name])

        # Create a new frame for parameters
        self.claims_parameter_frame = ttk.Frame(self)
        self.claims_parameter_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        # Generate input fields for claims distribution parameters
        self.claims_parameters = {}
        if hasattr(claims_distribution, 'shapes') and claims_distribution.shapes:
            param_names = claims_distribution.shapes.split(",")
            for i, param_name in enumerate(param_names):
                param_label = ttk.Label(self.claims_parameter_frame, text=f"Claims {param_name.strip()}")
                param_label.grid(row=i, column=0, padx=10, pady=5, sticky="w")

                param_entry = ttk.Entry(self.claims_parameter_frame)
                param_entry.grid(row=i, column=1, padx=10, pady=5)

                self.claims_parameters[param_name.strip()] = param_entry
        else:
            print(f"No parameters found for distribution '{claims_distribution_name}'.")

    def generate_severity_parameter_inputs(self, event=None):
        # Remove previous parameter frame
        if hasattr(self, "severity_parameter_frame"):
            self.severity_parameter_frame.destroy()

        severity_distribution_name = self.severity_distribution_var.get()
        severity_distribution = getattr(stats, self.severity_distributions[severity_distribution_name])

        # Create a new frame for parameters
        self.severity_parameter_frame = ttk.Frame(self)
        self.severity_parameter_frame.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        # Generate input fields for severity distribution parameters
        self.severity_parameters = {}

        if severity_distribution_name == "Lognormal":
            param_names = ["Shape", "Scale"]
        elif severity_distribution_name == "Weibull":
            param_names = ["Shape", "Scale"]
        elif severity_distribution_name == "Pareto":
            param_names = ["Shape", "Scale"]
        elif severity_distribution_name == "Frechet":
            param_names = ["Shape", "Scale"]
        elif severity_distribution_name == "Exponential":
            param_names = ["Rate"]
        elif severity_distribution_name == "Gamma":
            param_names = ["Shape", "Scale"]
        elif severity_distribution_name == "Normal":
            param_names = ["Mean", "Standard Deviation"]
        else:
            print(f"No parameters found for distribution '{severity_distribution_name}'.")
            return

        for i, param_name in enumerate(param_names):
            param_label = ttk.Label(self.severity_parameter_frame, text=f"Severity {param_name}")
            param_label.grid(row=i, column=0, padx=10, pady=5, sticky="w")

            param_entry = ttk.Entry(self.severity_parameter_frame)
            param_entry.grid(row=i, column=1, padx=10, pady=5)

            self.severity_parameters[param_name] = param_entry

    def calculate(self):
        claims_distribution_name = self.claims_distribution_var.get()
        severity_distribution_name = self.severity_distribution_var.get()

        claims_distribution = getattr(stats, self.claims_distributions[claims_distribution_name])
        severity_distribution = getattr(stats, self.severity_distributions[severity_distribution_name])

        claims_params = [float(param_entry.get()) for param_entry in self.claims_parameters.values()]
        severity_params = [float(param_entry.get()) for param_entry in self.severity_parameters.values()]

        # Perform calculations with chosen distributions and parameters
        # Example: You can compute tail probabilities here

        print("Claims Distribution:", claims_distribution_name)
        print("Claims Distribution Parameters:", claims_params)
        print("Severity Distribution:", severity_distribution_name)
        print("Severity Distribution Parameters:", severity_params)

        doCalculations(claims_distribution_name, claims_params, severity_distribution_name, severity_params)


if __name__ == "__main__":
    app = DistributionCalculator()
    app.mainloop()
