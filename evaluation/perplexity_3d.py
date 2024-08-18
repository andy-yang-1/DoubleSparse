import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import json

# with open("3d-bar-ppl.json", "r") as f:
with open("3d-bar-ppl-mistral.json", "r") as f:
    wiki_2_perplexity_results = json.load(f)

plt.rc('font', size=12)  # 默认字体大小12，加粗
fig = plt.figure(figsize=(7, 7))
# fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set the positions for the ticks to be at 0, 1, 2, 3, etc.
sparsity_levels = [1, 2, 4, 8, 16, 32, 64, 128]
positions = list(range(len(sparsity_levels)))

# Set the labels to be 1, 2, 4, 8, etc.
labels = [f"1/{i}" if i !=1 else 1 for i in sparsity_levels]

ax.set_xticks(positions)
ax.set_xticklabels(labels)
ax.set_yticks(positions)
ax.set_yticklabels(labels)

ax.set_xlabel('Token Sparsity')
ax.set_ylabel('Channel Sparsity')
ax.set_zlabel('Wiki-2 Score')

# for model_path, model_results in wiki_2_perplexity_results.items():
# for config_name, perplexity in wiki_2_perplexity_results["meta-llama/Llama-2-7b-hf"].items():
for config_name, perplexity in wiki_2_perplexity_results["mistralai/Mistral-7B-v0.1"].items():
    parts = config_name.split('-')
    if "heavy" in parts and "group" in parts:
        heavy_index = parts.index("heavy") + 1
        group_index = parts.index("group") + 1
        heavy_const = int(parts[heavy_index])
        group_factor = int(parts[group_index])
        qbit = int(parts[1])
        x = np.log2(2048 // heavy_const)
        y = np.log2(group_factor * 16 // qbit)
        if x == 4 and y == 4:
            color = 'r'
        else:
            color = 'g'

        if np.isnan(perplexity) or perplexity > 10:
            z = 0
        else:
            z = 10 - perplexity
        if np.isnan(perplexity):
            print(f"perplexity is nan for {config_name}, z = {z}")
        if perplexity > 10:
            print(f"perplexity is {perplexity} for {config_name}, z = {z}")

        # z = -perplexity if perplexity <= 10 else -10
        ax.bar3d(x, y, 0, 1, 1, z, shade=True,color=color)

ax.set_title('Mistral-7B-v0.1')
# ax.set_title('Llama-2-7b-hf')
# ax.set_zlim(0, 6)
ax.view_init(elev=30, azim=30)

plt.tight_layout()

plt.savefig("data/3d/Mistral-7B-v0.1.png")
# plt.savefig("data/3d/Llama-2-7b-hf.png")
