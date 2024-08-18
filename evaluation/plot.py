import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# Function to save the plots to the local filesystem
def save_perplexity_plots(data_json):
    # Load JSON data
    data = json.loads(data_json)

    # Create a directory for the plots
    output_dir = "./data/ppl/"
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through each model in the data
    for model, values in data.items():
        # Separate data for qbit-8 and qbit-4
        qbit_8_data = {k: v for k, v in values.items() if 'qbit-8' in k}
        qbit_4_data = {k: v for k, v in values.items() if 'qbit-4' in k}

        # Prepare plot data for q, k, qk for both qbit-8 and qbit-4
        for qbit_data, bit_num in [(qbit_8_data, 8), (qbit_4_data, 4)]:
            if bit_num == 8:
                sparsity_levels = [1, 2, 4, 8, 16, 32]
            else:
                sparsity_levels = [1, 4, 8, 16, 32]
            # sparsity_levels = [1, 2, 4, 8, 16, 32]
            q_values = [qbit_data.get(f'qbit-{bit_num}-sparsity-{s}-q', None) for s in sparsity_levels]
            k_values = [qbit_data.get(f'qbit-{bit_num}-sparsity-{s}-k', None) for s in sparsity_levels]
            qk_values = [qbit_data.get(f'qbit-{bit_num}-sparsity-{s}-qk', None) for s in sparsity_levels]
            q_values[0] = values.get('original', None)
            k_values[0] = values.get('original', None)
            qk_values[0] = values.get('original', None)
            print(f"qbit_data: {qbit_data}")
            print(f"q_values: {q_values}")
            # Create the plot
            plt.figure(figsize=(10, 6))
            plt.plot(sparsity_levels, q_values, label='q', marker='o')
            plt.plot(sparsity_levels, k_values, label='k', marker='s')
            plt.plot(sparsity_levels, qk_values, label='qk', marker='^')

            # Set plot attributes
            title = f"{model}-{bit_num}-bit"
            plt.title(f"{title}")
            plt.xlabel('Sparsity Level')
            plt.ylabel('Perplexity')
            plt.ylim(5, 50)

            plt.xscale('log')
            plt.gca().xaxis.set_major_locator(ticker.LogLocator(base=2))

            plt.xticks(sparsity_levels, sparsity_levels)  # Set x-axis ticks to sparsity levels
            plt.grid(True)
            plt.legend()


            # Save the plot
            filename = f"{output_dir}/{title.replace('/', '_').replace(' ', '_')}.png"
            plt.savefig(filename)
            plt.close()

    return output_dir

# Assuming data_json is a valid JSON string
# data_json = "your JSON data here"

# Call the function with the provided JSON data
# output_dir = save_perplexity_plots(data_json)

# The function will save the plots in the specified directory and return the directory path.
# Note: The above function call is commented out because 'data_json' is not defined.
# Replace 'your JSON data here' with your actual JSON data string to use this function.

with open("wiki-2-ppl.json", "r") as f:
    data_json = f.read()

output_dir = save_perplexity_plots(data_json)