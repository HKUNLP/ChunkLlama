import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def draw(output_name):
    acc_list = []
    # Iterating through each file and extract the 3 columns we need
    with open(output_name, 'r') as f:
        for line in f:
            json_data = json.loads(line)
            # Extracting the required fields
            document_depth = json_data.get("depth_percent", None)
            context_length = json_data.get("context_length", None)
            score = json_data.get("score", None)
            # Appending to the list
            acc_list.append({
                "Document Depth": document_depth,
                "Context Length": context_length,
                "score": score
            })
    # Creating a DataFrame
    df = pd.DataFrame(acc_list)
    vmin, vmax = 0.00, 2.00
    from matplotlib.colors import LinearSegmentedColormap
    # Create the pivot table
    pivot_table = pd.pivot_table(df, values='score', index=['Document Depth'], columns=['Context Length'],
                                 aggfunc='mean')
    # Create a custom colormap for the heatmap
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

    # Create the heatmap with the custom normalization
    plt.figure(figsize=(8, 4))  # Adjust these dimensions as needed
    sns.heatmap(
        pivot_table,
        cmap=cmap,
        vmin=vmin,  # Use the TwoSlopeNorm object
        cbar_kws={'label': 'score'}
        # Uncomment the following line if you want to annotate cells with their numerical value
        # annot=True
    )
    # More aesthetics
    plt.title(title)  # Adds a title, uncomment to use
    plt.xlabel('Token Limit')  # X-axis label
    plt.ylabel('Depth Percent')  # Y-axis label
    plt.xticks(rotation=0)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area
    # Show the plot
    plt.savefig(output_name.replace("jsonl", "png"))
    plt.show()

title = 'Pressure Testing Chunk-Mistral 192k Context\n("Needle In A HayStack")'
input_file_name = "mistral.output.jsonl"
draw(output_name=input_file_name)
