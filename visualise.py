import matplotlib.pyplot as plt
import mpld3
import napari

def create_interactive_html_plot(x_data, y_data, title="Interactive Plot", x_label="X-axis", y_label="Y-axis"):
    """
    Creates an interactive HTML plot using matplotlib and mpld3 libraries.

    :param x_data: Data for the X-axis.
    :param y_data: Data for the Y-axis.
    :param title: Title of the plot.
    :param x_label: Label for the X-axis.
    :param y_label: Label for the Y-axis.
    :return: HTML representation of the plot.
    """
    
    # Create a new matplotlib figure and axis.
    fig, ax = plt.subplots()

    # Plot the data.
    ax.plot(x_data, y_data, linestyle='-', marker='o')

    # Set the title and labels.
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Enable grid.
    ax.grid(True)

    # Convert to HTML using mpld3.
    interactive_html = mpld3.fig_to_html(fig)

    # Optionally, save the figure to an HTML file.
    # mpld3.save_html(fig, "plot.html")
    
    # Clean up plt to avoid excessive memory use.
    plt.close(fig)

    # Return the HTML.
    return interactive_html

def visualize_data(data):
    viewer = napari.Viewer()
    viewer.add_image(data)
    napari.run()
