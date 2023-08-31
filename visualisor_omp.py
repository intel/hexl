import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd

filename = "hexl_omp_out_0824_23_59.csv"

data = {}

with open(filename, 'r') as file:
    reader = csv.reader(file)
    
    input_size = None
    
    for row_index, row in enumerate(reader):
        if len(row) == 0:
            continue

        print(f"Debug: Processing row {row_index}: {row}")  # Debug print

        if row[0].startswith("Input Size"):
            input_size = row[0].split('=')[1].strip()
            data[input_size] = {}
            next(reader)  # Skip headers
        else:
            row_split = row[0].split()
            method = row_split[0]
            print(f"Debug: method is: {method}")  # Debug print

            for i, threads in enumerate(["Threads=1", "Threads=2", "Threads=4", "Threads=6", "Threads=8"]):
                
                print(f"Debug: i = {i}, row length = {len(row_split)}")  # Debug print

                if i + 1 >= len(row_split):
                    print(f"Warning: Skipping index {i + 1} as it's out of range for row {row}")
                    continue

                if not method in data[input_size]:
                    data[input_size][method] = {}
                
                print(f"row_split[i+1] = {row_split[i+1]}")  # Debug print
                data[input_size][method][threads] = float(row_split[i + 1])


serial_times = {}

# Read the file with serial execution times
with open("hexl_ser_out_0824_1431.csv", 'r') as file:
    lines = file.readlines()
    # Extract header to get input sizes
    header = lines[0].strip().split()
    input_sizes = [int(input_size.split('=')[1]) for input_size in header[1:]]

    # Loop through each row (skipping the header)
    for line in lines[1:]:
        elements = line.strip().split()
        method = elements[0]
        times = [float(time) for time in elements[1:]]
        serial_times[method] = {}
        for input_size, time in zip(input_sizes, times):
            serial_times[method][input_size] = time

# print(serial_times)


for method in data[list(data.keys())[0]].keys():  # Assuming each input size has the same methods

    # Create a figure and layout for subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 10), gridspec_kw={'width_ratios': [1, 1, 1], 'height_ratios': [1, 1]})
    fig.suptitle(f'{method} - Execution Time vs Thread Count')

    axs = axs.flatten()

    # First set of graphs for each method: Execution time vs Thread number
    for ax, (input_size, input_data) in zip(axs, data.items()):
        thread_counts = []
        times = []
        for thread, time in input_data[method].items():
            thread_counts.append(int(thread.split("=")[1]))
            times.append(time)

        ax.plot(thread_counts, times, label=f"Input Size {input_size}")

        ax.axhline(y=serial_times[method][int(input_size)], color='r', linestyle='--', label='Serial input size {input_size}')

        ax.set_xlabel('Thread Count')
        ax.set_ylabel('Execution Time')
        ax.set_title(f'Input Size = {input_size}')
        ax.grid(True)

    axs[-1].set_visible(False)

    # Add a single legend for all subplots
    fig.legend(loc="lower right", bbox_to_anchor=(0.9, 0.1))

    # Add a text caption below the figure
    # fig.text(0.5, -0.05, 'Caption: Description of the figure.', ha='center')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Save the figure
    plt.savefig(f"Dissertation/{method}_Execution_Time_vs_Thread_Count.png")

    # plt.show()

 
num_methods = len(data[list(data.keys())[0]].keys())
fig, axs = plt.subplots(9, 1, figsize=(12, 20))
# fig.suptitle('Execution Time vs Input Size')

# Flatten axes array if it's multidimensional
if num_methods > 1:
    axs = axs.flatten()

# Loop through each method
for ax, method in zip(axs, data[list(data.keys())[0]].keys()):
    for thread in ["Threads=1", "Threads=2", "Threads=4", "Threads=6", "Threads=8"]:
        input_sizes = []
        times = []
        for input_size in data.keys():
            input_sizes.append(int(input_size))
            times.append(data[input_size][method][thread])

        ax.plot(input_sizes, times, label=f"{thread}")

    ax.set_xlabel('Input Size')
    ax.set_ylabel('Execution Time(s)')
    ax.set_title(f'{method}')
    if ax == 0:
        ax.legend(loc="lower right")

    ax.set_xscale('log', base=2)
    ax.set_xticks([2**12, 2**16, 2**20, 2**24, 2**28])
    ax.set_xticklabels(['2^12', '2^16', '2^20', '2^24', '2^28'])
    ax.set_yscale('log', base=2)
    ax.grid(True)



# Tighten layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# Save the figure
plt.savefig("Dissertation/Execution_Time_vs_Input_size.png")
# Show or save figure
# plt.show()





