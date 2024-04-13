import matplotlib.pyplot as plt
import torch.nn as nn
from torch import Tensor, rand, pi, sin, zeros_like, no_grad


def plot(xs, actual_ys, predicted_ys, x_label, y_label, title, subtitle, path):
    "Based on https://towardsdatascience.com/5-steps-to-build-beautiful-line-charts-with-python-655ac5477310"

    fig, ax = plt.subplots(figsize=(13.33, 7.5), dpi=96)

    if actual_ys is not None:
        ax.plot(xs, actual_ys, 'o', zorder=2, label='Actual y=sin(x)')
    if predicted_ys is not None:
        ax.plot(xs, predicted_ys, 'h', zorder=2, label='Predicted y=sin(x)')

    # Create the grid
    ax.grid(which="major", axis='x', color='#DAD8D7', alpha=0.5, zorder=1)
    ax.grid(which="major", axis='y', color='#DAD8D7', alpha=0.5, zorder=1)

    # Reformat x-axis label and tick labels
    ax.set_xlabel(x_label, fontsize=12, labelpad=10)  # No need for an axis label
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.set_tick_params(pad=2, labelbottom=True, bottom=True, labelsize=12, labelrotation=0)

    # Reformat y-axis
    ax.set_ylabel(y_label, fontsize=12, labelpad=10)
    ax.yaxis.set_label_position("left")
    ax.yaxis.set_tick_params(pad=2, labeltop=False, labelbottom=True, bottom=False, labelsize=12)

    # Remove the spines
    ax.spines[['top', 'right']].set_visible(False)

    # Make the left and bottom spine thicker
    ax.spines['bottom'].set_linewidth(1.1)
    ax.spines['left'].set_linewidth(1.1)

    # Add in red line and rectangle on top
    ax.plot([0.05, .9], [.98, .98], transform=fig.transFigure, clip_on=False, color='#FFD700', linewidth=.6)
    ax.add_patch(plt.Rectangle((0.05, .98), 0.04, -0.02, facecolor='#FFD700',
                               transform=fig.transFigure, clip_on=False, linewidth=0))

    # Add in title and subtitle
    ax.text(x=0.05, y=.93, s=title, transform=fig.transFigure, ha='left', fontsize=14, weight='bold', alpha=.8)
    ax.text(x=0.05, y=.90, s=subtitle, transform=fig.transFigure, ha='left', fontsize=12, alpha=.8)

    # Adjust the margins around the plot area
    plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.85, wspace=None, hspace=None)

    # Set a white background
    fig.patch.set_facecolor('white')

    # Add legend
    if actual_ys is not None and predicted_ys is not None:
        plt.legend()

    if path:
        plt.savefig(path)


def plot_prediction(nnet: nn.Module, training_size: int):
    n: int = 1000
    xs: Tensor = (2 * pi - 0) * rand(n)
    actual_ys: Tensor = sin(xs)
    predicted_ys: Tensor = zeros_like(xs)

    # Set model to evaluation mode
    nnet.eval()

    with no_grad():
        for i, x in enumerate(xs):
            # Get the prediction from the model
            predicted_ys[i] = nnet(Tensor([x]))

    subtitle: str = f'Trained on {training_size:,} Values & Plotted on {n:,} Values'
    plot(xs=xs, actual_ys=actual_ys, predicted_ys=None, x_label='x', y_label='Actual sin(x)',
         title=f'x vs. Actual sin(x)', subtitle=subtitle,
         path=f'./plots/x_vs_actual_ys_{training_size}')

    plot(xs=xs, actual_ys=None, predicted_ys=predicted_ys, x_label='x', y_label='Predicted sin(x)',
         title=f'x vs. Predicted sin(x)', subtitle=subtitle,
         path=f'./plots/x_vs_pred_ys_{training_size}')

    plot(xs, actual_ys, predicted_ys,'x','sin(x)',
         title=f'Actual and Predicted x vs. sin(x)', subtitle=subtitle,
         path=f'./plots/xs_vs_ys_all_{training_size}')
