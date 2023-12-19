import matplotlib.pyplot as plt
import pandas as pd

def visualize_data(data, y):
    # Plotting the PCA results
    plt.figure(figsize=(8, 6))
    plt.title('PCA Visualization')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    # Transform the data into a data frame
    column_names = ['principal_component_1', 'principal_component_2']
    df = pd.DataFrame(data=data, columns=column_names)
    
    targets = ['M', 'B']
    colors = ['r', 'g']
    
    for target, color in zip(targets, colors):
        indices_to_keep = y == target
        plt.scatter(
            df.loc[indices_to_keep, 'principal_component_1'],
            df.loc[indices_to_keep, 'principal_component_2'],
            c=color,
            alpha=0.7,
            label=target
        )
    plt.legend()
    plt.show()