# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression

# # Read the CSV file into a pandas DataFrame
# df = pd.read_csv('hexl_ser_out_0824_1431.csv', delim_whitespace=True)

# # Strip leading and trailing whitespaces from column names
# df.columns = df.columns.str.strip()

# # Inspect the DataFrame
# print(df.head())

# # Check if 'Method' column exists
# try:
#     print(df['Method'])
# except KeyError:
#     print("Column 'Method' not found!")

# # Input sizes from the DataFrame columns
# input_sizes = [int(col.split('=')[1]) for col in df.columns if 'Input_size' in col]
# x_vals = np.log2(input_sizes).reshape(-1, 1)  # Log-transform the x-values

# # Enlarge the figure size
# plt.figure(figsize=(15,21))

# # Loop over each method
# for idx, method in enumerate(df['Method']):
#     y_vals = np.log(df.loc[idx, 'Input_size=4096':'Input_size=268435456'].to_numpy(dtype='float64'))  # Log-transform the y-values


#     # Linear Regression
#     model = LinearRegression()
#     model.fit(x_vals, y_vals)
#     x_test = np.linspace(min(x_vals), max(x_vals), 300).reshape(-1, 1)
#     y_pred = model.predict(x_test)

#     # Calculate residuals
#     residuals = y_vals - model.predict(x_vals)

#     # Calculate the standard deviation of the residuals
#     residual_std = np.std(residuals)

#     # Define the color for this particular method
#     current_color = next(plt.gca()._get_lines.prop_cycler)['color']

#     # Plotting
#     plt.scatter(x_vals, y_vals, label=f"{method} (slope: {model.coef_[0]:.2f})", color=current_color)
#     plt.plot(x_test, y_pred, color=current_color)

#     plt.errorbar(x_vals.flatten(), y_vals, yerr=residual_std, fmt='o', capsize=5, color=current_color)


# plt.xticks(x_vals.flatten(), [f'$2^{{{int(x)}}}$' for x in x_vals.flatten()], fontsize='20')

# # Move the legend to the top left within the graph
# plt.legend(loc='upper left', fontsize='20')

# # Add labels and grid
# plt.xlabel('Input Size',fontsize='20')
# plt.ylabel('Logarithm to base 2 of Execution Time (ms)',fontsize='20')
# plt.grid(True)

# # Adjust layout and show the plot
# plt.tight_layout()
# # Save the figure as a high-resolution jpg file
# plt.savefig('hexl_ser_plot.jpg', format='jpg', dpi=300, bbox_inches='tight', pad_inches=0.1)

# plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import math

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('hexl_ser_out_0824_1431.csv', delim_whitespace=True)
df.columns = df.columns.str.strip()

# Input sizes from the DataFrame columns
input_sizes = [int(col.split('=')[1]) for col in df.columns if 'Input_size' in col]
x_vals = np.log2(input_sizes).reshape(-1, 1)

# Create polynomial features
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x_vals)

# Enlarge the figure size
plt.figure(figsize=(15, 21))

# Loop over each method
for idx, method in enumerate(df['Method']):
    y_vals = np.log(df.loc[idx, 'Input_size=4096':'Input_size=268435456'].to_numpy(dtype='float64'))

    color = next(plt.gca()._get_lines.prop_cycler)['color']


    if method == "BM_NTTInPlace":
        # Polynomial Regression
        model = LinearRegression()
        model.fit(x_poly, y_vals)
        poly_y_pred = model.predict(x_poly)

        # Calculate the standard error of the residuals
        mse = mean_squared_error(y_vals, poly_y_pred)
        std_error = math.sqrt(mse)

        # Generate test data and predict
        x_test = np.linspace(min(x_vals), max(x_vals), 300).reshape(-1, 1)
        x_test_poly = poly.transform(x_test)
        y_pred_test = model.predict(x_test_poly)

        plt.scatter(x_vals, y_vals, color=color, label=f"{method} (coef: {model.coef_[1]:.2f}, {model.coef_[2]:.2f})")
        plt.errorbar(x_vals, y_vals, yerr=std_error, fmt='o', color=color)
        plt.plot(x_test, y_pred_test, color=color)
    
    else:
        # Linear Regression
        model = LinearRegression()
        model.fit(x_vals, y_vals)
        x_test = np.linspace(min(x_vals), max(x_vals), 300).reshape(-1, 1)
        y_pred = model.predict(x_test)

        # Calculate residuals
        residuals = y_vals - model.predict(x_vals)

        # Calculate the standard deviation of the residuals
        residual_std = np.std(residuals)

    # Plotting
        plt.scatter(x_vals, y_vals, label=f"{method} (slope: {model.coef_[0]:.2f})", color=color)
        plt.plot(x_test, y_pred, color=color)
        plt.errorbar(x_vals.flatten(), y_vals, yerr=residual_std, fmt='o', capsize=5, color=color)

# Move the legend to the top left within the graph
plt.legend(loc='upper left', fontsize='20')

# Set custom x-ticks
plt.xticks(x_vals.flatten(), [f'$2^{{{int(x)}}}$' for x in x_vals.flatten()], fontsize='20')

# Add labels and grid
plt.xlabel('Input Size', fontsize='20')
plt.ylabel('Logarithm to base 2 of Execution Time (ms)', fontsize='20')
plt.grid(True)

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig('hexl_ser_plot_poly_with_error.jpg', format='jpg', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
