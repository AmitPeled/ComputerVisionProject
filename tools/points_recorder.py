import matplotlib.pyplot as plt
import pickle
import os
import numpy as np

# List to store clicked points
clicked_points = []

# File where points will be saved
points_file = 'clicked_points.pkl'

# Maximum distance to consider a point for deletion (in data coordinates)
max_distance = 0.5


def on_click(event):
    if event.button == 1:  # Left-click to add points
        if event.xdata is not None and event.ydata is not None:  # Ignore clicks outside the plot
            clicked_points.append((event.xdata, event.ydata))
            plt.plot(event.xdata, event.ydata, 'ro')  # 'ro' means red dots
            plt.draw()

            # Save the clicked points after each click
            save_points()

    elif event.button == 3:  # Right-click to delete points
        if event.xdata is not None and event.ydata is not None:  # Ignore clicks outside the plot
            remove_point(event.xdata, event.ydata)
            redraw_plot()  # Redraw the points after removal
            plt.draw()
            save_points()  # Save the updated points


def save_points():
    with open(points_file, 'wb') as f:
        pickle.dump(clicked_points, f)


def load_points():
    if os.path.exists(points_file):
        with open(points_file, 'rb') as f:
            return pickle.load(f)
    return []


def remove_point(x, y):
    global clicked_points
    # Find the closest point within the maximum distance
    for point in clicked_points:
        if np.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2) < max_distance:
            clicked_points.remove(point)
            break


def redraw_plot():
    # Clear the figure and re-plot the points with fixed axis limits
    plt.clf()  # Clear the current figure
    ax = plt.gca()  # Get the current axes
    ax.set_xlim(0, 10)  # Set fixed x-axis limit
    ax.set_ylim(0, 10)  # Set fixed y-axis limit
    display_points()  # Replot the remaining points


def display_points():
    print(clicked_points)
    # Plot all the points that are still in clicked_points
    for point in clicked_points:
        plt.plot(point[0], point[1], 'ro')  # Plot the saved points as red dots


# Create the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(0, 10)  # Fixed x-axis limit
ax.set_ylim(0, 10)  # Fixed y-axis limit
ax.set_title("Click to place points (Left-click) or remove points (Right-click)")

# Load previously clicked points if any
clicked_points = load_points()

# Plot the previously saved points
display_points()

# Connect the click event to the handler
fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()
