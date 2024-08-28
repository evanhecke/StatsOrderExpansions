import tkinter as tk
import matplotlib.pyplot as plt
from tkinter import ttk
from approximations import doCalculations
from PIL import Image, ImageTk
import numpy as np
import io


class PlottingWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Plotting Window")
        self.configure(bg='white')

        # Initial dimensions
        self.geometry("1200x600")

        # Create frames for layout
        self.left_frame = ttk.Frame(self, width=300, relief=tk.SUNKEN)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10, anchor='n')

        self.right_frame = ttk.Frame(self)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Canvas for smaller plots with vertical scrollbar
        self.smaller_plots_canvas = tk.Canvas(self.left_frame, bg='white')
        self.smaller_plots_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar = ttk.Scrollbar(self.left_frame, orient="vertical", command=self.smaller_plots_canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.smaller_plots_canvas.configure(yscrollcommand=self.scrollbar.set)

        # Frame inside canvas for smaller plots
        self.smaller_plots_frame = ttk.Frame(self.smaller_plots_canvas)
        self.smaller_plots_canvas.create_window((0, 0), window=self.smaller_plots_frame, anchor="nw")

        # Label for displaying the large plot
        self.large_plot_label = ttk.Label(self.right_frame)
        self.large_plot_label.pack(fill=tk.BOTH, expand=True)

        self.plot_images = []
        self.plot_map = {}

        # Bind resize event to update canvas scrollregion
        self.bind("<Configure>", self.on_resize)

        # Variable to keep track of the currently displayed plot
        self.current_plot_index = None
        self.current_plot_title = None  # To keep track of the current plot title

    def on_resize(self, event):
        """Update scrollregion to match the canvas contents."""
        self.smaller_plots_canvas.config(scrollregion=self.smaller_plots_canvas.bbox("all"))

    def update_window_size(self, width, height):
        """Set the size of the window dynamically."""
        self.geometry(f"{width}x{height}")

    def add_smaller_plot(self, s_values, *results, title):
        """Add a smaller plot to the left frame with a given title."""
        fig, ax = plt.subplots(figsize=(3, 2))  # Smaller size for smaller plots
        for i, result in enumerate(results, start=1):
            if result is not None:
                ax.plot(s_values, result, label=f"Result {i}")

        ax.set_xlabel("s")
        ax.set_ylabel("Relative Approximations")
        ax.set_title(title)  # Use the provided title

        # Set custom axis scales
        ax.set_ylim(-2, 12)  # Adjust these limits as needed
        ax.set_xscale('log')  # Set x-axis to logarithmic scale

        # Save plot to BytesIO stream
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)
        image = Image.open(buf)
        photo = ImageTk.PhotoImage(image)

        # Store plot for later selection
        index = len(self.plot_images)
        self.plot_images.append(photo)
        self.plot_map[index] = (photo, s_values, *results, title)

        # Display smaller plot in canvas
        y_position = index * 250  # Adjust spacing as needed
        plot_id = self.smaller_plots_canvas.create_image(150, y_position, image=photo, tags=str(index))
        self.smaller_plots_frame.update_idletasks()

        # Update scrollregion
        self.smaller_plots_canvas.config(scrollregion=self.smaller_plots_canvas.bbox("all"))

        # Bind click event
        self.smaller_plots_canvas.tag_bind(plot_id, "<Button-1>", lambda e, idx=index: self.on_plot_click(idx))

        plt.close(fig)

    def show_large_plot(self, s_values, *results, title):
        """Display the large plot in the right frame with a given title."""
        fig, ax = plt.subplots(figsize=(8, 6))

        for i, result in enumerate(results, start=1):
            if result is not None:
                ax.plot(s_values, result, label=f"{i}-th Order Approximation")
            else:
                break # Break out of the for loop as soon as i-th order approximation doesn't exist
        ax.set_xlabel("s")
        ax.set_ylabel("Relative Approximations")
        ax.set_title(title)  # Use the provided title
        ax.legend()

        # Set custom axis scales
        ax.set_ylim(-2, 12)  # Adjust these limits as needed
        ax.set_xscale('log')  # Set x-axis to logarithmic scale

        # Save to BytesIO stream
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)
        image = Image.open(buf)
        photo = ImageTk.PhotoImage(image)

        # Display large plot
        self.large_plot_label.config(image=photo)
        self.large_plot_label.image = photo  # Keep a reference

        plt.close(fig)

    def on_plot_click(self, index):
        """Handle clicks on smaller plots to show the corresponding large plot and swap the large plot back to small."""
        if index in self.plot_map:
            # Get the currently clicked plot's data
            clicked_plot_data = self.plot_map[index]
            _, s_values, *results, title = clicked_plot_data

            # Show this plot as the large plot
            self.show_large_plot(s_values, *results, title=title)

            # Set the clicked plot as the current large plot
            self.current_plot_index = index
            self.current_plot_title = title  # Update current plot title



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

        self.plotting_window = None  # Placeholder for the plotting window

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

        # Frame for plotting
        self.plot_frame = ttk.Frame(self)
        self.plot_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")

        # Initial plot placeholder
        self.plot_label = ttk.Label(self.plot_frame)
        self.plot_label.pack()

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

        # Get data for plotting
        s_values, results1, results2, results3, results4, results5, results6 = doCalculations(
            claims_distribution_name, claims_params, severity_distribution_name, severity_params
        )

        # Pass distribution details to plot_results
        self.plot_results(
            s_values, claims_distribution_name, claims_params,
            severity_distribution_name, severity_params,
            results1, results2, results3, results4, results5, results6,
        )

    def count_sign_changes(self, result):
        """Count the number of times the sign changes in the result."""
        if result is None or len(result) < 2:
            return 0

        sign_changes = 0
        previous_sign = np.sign(result[0])

        for value in result[400:1000]: # change ending value accordingly
            current_sign = np.sign(value)
            if current_sign != previous_sign and current_sign != 0:
                sign_changes += 1
                previous_sign = current_sign

        return sign_changes

    def plot_results(self, s_values, claims_dist_name, claims_params, severity_dist_name, severity_params, *results):
        """Open or update the plotting window with new results and titles."""
        if not self.plotting_window:
            self.plotting_window = PlottingWindow(self)

        # Generate titles based on the chosen distributions and parameters
        claims_title = f"{claims_dist_name}({', '.join(f'{v}' for v in claims_params)})"
        severity_title = f"{severity_dist_name}({', '.join(f'{v}' for v in severity_params)})"
        plot_title = f"{claims_title}, {severity_title}"

        # Prepare results for plotting
        results_to_plot = []
        oscillatory_results = []

        # Check of the weibull approximations don't oscillate too much
        for i in range(len(results)):
            if results[i] is not None:
                if severity_dist_name == "Weibull":
                    result = results[i]
                    sign_changes = self.count_sign_changes(result)
                    if i >= 2 and sign_changes > 20:
                        # This is a higher-order approximation that is oscillatory
                        oscillatory_results.append(f"{i + 1}-th Order Approximation")
                    else:
                        # This result is not oscillatory and should be plotted
                        results_to_plot.append(result)
                else:
                    result = results[i]
                    results_to_plot.append(result)

        # Check if any higher-order approximations are missing
        missing_approximations = [f"{i + 1}-th Order Approximation" for i in range(2, 6) if i >= len(results) or results[i] is None]
        if missing_approximations and severity_dist_name != "Frechet":
            missing_msg = ("Some higher order approximations don't exist for the chosen parameter values:\n" +
                           "\n".join(missing_approximations))
            tk.messagebox.showwarning("Missing Approximations", missing_msg)
        elif severity_dist_name == "Frechet":
            missing_msg = ("Some higher order approximations are not shown because the calculation takes too long for the Frechet distribution:\n" +
                           "\n".join(missing_approximations))
            tk.messagebox.showwarning("Missing Approximations", missing_msg)

        if oscillatory_results and severity_dist_name == "Weibull":
            oscillation_msg = ("The following higher order approximations are too oscillatory and will "
                               "not be plotted:\n") + "\n".join(oscillatory_results)
            tk.messagebox.showwarning("Oscillatory Results", oscillation_msg)

        # Plot results
        if results_to_plot:
            self.plotting_window.show_large_plot(s_values, *results_to_plot, title=plot_title)
            self.plotting_window.add_smaller_plot(s_values, *results_to_plot, title=plot_title)
        else:
            tk.messagebox.showwarning("No Results", "No results available to plot.")


if __name__ == "__main__":
    app = DistributionCalculator()
    app.mainloop()
