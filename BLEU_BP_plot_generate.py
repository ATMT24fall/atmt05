import matplotlib.pyplot as plt

# Example data
beam_sizes = [1, 5, 10, 15, 18, 20, 23, 25]  # Replace with your beam sizes
bleu_scores = [
    15.9,
    19.3,
    20.8,
    21.4,
    21.6,
    21.7,
    21.2,
    21.0,
]  # Replace with BLEU scores
brevity_penalties = [
    1.000,
    1.000,
    1.000,
    1.000,
    0.992,
    0.977,
    0.932,
    0.899,
]  # Replace with brevity penalties

# Create the plot
fig, ax1 = plt.subplots()

# Plot BLEU scores
ax1.set_xlabel("Beam Size")
ax1.set_ylabel("BLEU Score", color="blue")
ax1.plot(beam_sizes, bleu_scores, marker="o", color="blue", label="BLEU Score")
ax1.tick_params(axis="y", labelcolor="blue")

# Plot brevity penalties on a secondary axis
ax2 = ax1.twinx()
ax2.set_ylabel("Brevity Penalty", color="red")
ax2.plot(
    beam_sizes,
    brevity_penalties,
    marker="x",
    color="red",
    linestyle="--",
    label="Brevity Penalty",
)
ax2.tick_params(axis="y", labelcolor="red")

# Add grid and legends
fig.tight_layout()
plt.title("BLEU Score and Brevity Penalty over Beam Size")

# Save the plot as a file
plt.savefig("bleu_vs_brevity_penalty.png")  # Change the file name and format as needed
plt.show()
