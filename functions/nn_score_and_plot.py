import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
mean_error = lambda y_pred, y_true: np.mean(y_pred.reshape(y_true.shape)-y_true)

def fit_linear(obs, pred):
    """Fit pred = slope * obs + intercept via OLS and return slope, intercept, residuals."""
    model = LinearRegression()
    model.fit(np.array(obs).reshape(-1, 1), np.array(pred))
    slope     = model.coef_[0]
    intercept = model.intercept_
    residuals = np.array(pred) - (slope * np.array(obs) + intercept)
    return slope, intercept, residuals

def calculate_score(obs, pred):
    
    r2 = r2_score(obs, pred)

    # sklearn Linear Regression (pred = slope * obs + intercept)
    model = LinearRegression()
    model.fit(np.array(obs).reshape(-1, 1), pred)   # fit(X, y)

    slope = model.coef_[0]
    intercept = model.intercept_
    
    scores = [r2, 
              root_mean_squared_error(pred, obs), 
              mean_absolute_error(pred, obs), 
              mean_error(pred, obs),
              slope,
              intercept]
    
    return np.array(scores)

def plot_training_loss(train_losses):
    fontsize=14
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.xlabel('Epoch', fontsize=fontsize)
    plt.ylabel('Loss', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title('Training Loss', fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.grid(True)

def set_tick_labels(ax, ytick_size=12, xtick_mn_size=12, xtick_mj_size=12, rotation=45):
    ax.yaxis.set_major_formatter(ticker.EngFormatter())
    
    def custom_date_formatter(x, pos):
        dt = mdates.num2date(x)
        return dt.strftime('%b\n%Y')
    
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: mdates.num2date(x).strftime('\n%Y')))
    ax.xaxis.set_minor_formatter(ticker.FuncFormatter(lambda x, pos: mdates.num2date(x).strftime('%b')))
    
    ax.tick_params(axis='y', labelsize=ytick_size)

    ax.tick_params(axis='x', which='minor', length=3, rotation=rotation, labelsize=xtick_mn_size)
    ax.tick_params(axis='x', which='major', length=10, labelsize=xtick_mj_size)
    

def plot_time_series(df, 
                     cols_to_plot, 
                     ax,
                     loc='upper right', 
                     title='',
                     ylim=None,
                     xlim=(pd.Timestamp('2019-01-01'), pd.Timestamp('2021-12-31')),
                     show_xaxis=True
                    ):

    """ cols_to_plot: dict, key is integer, values are list of [column, label, color] """
    
    lw=1; fz = 14
    for i in range(len(cols_to_plot)):
        ax.plot(df['date'], 
                df[cols_to_plot[i][0]], 
                label     = cols_to_plot[i][1],
                color     = cols_to_plot[i][2], 
                linewidth = lw, 
                )

    if show_xaxis:
        set_tick_labels(ax, ytick_size=fz, xtick_mn_size=fz, xtick_mj_size=fz, rotation=55)
        ax.set_xlabel('Year', fontsize=fz)

    else:
        ax.set_xlabel('')
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        ax.tick_params(axis='y', labelsize=fz)
    
    ax.set_title(title, fontsize=18)
    ax.set_ylabel('PM$_{2.5}$ Concentrations \n ($\\mu$g$\\cdot$m$^{-3}$)', fontsize=fz)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    ncol = 2 if len(cols_to_plot) > 3 else 1
    ax.legend(ncol=ncol, 
              columnspacing=0.5 if ncol > 1 else 1,
              loc=loc, 
              fontsize=fz)
    ax.grid(True, linestyle='dotted')


def plot_scatter(ax, obs, pred, axmin=-20, axmax=400, fontsize=16, title='', show_xaxis=True, show_yaxis=True):
    slope, intercept, residuals = fit_linear(obs, pred)
    
    ax.scatter(obs, pred, alpha=0.5, s=5, color='black', label='Predictions') # scatter(x, y)
    ax.plot([axmin, axmax],
            [axmin, axmax], 'k--', lw=1, label='Perfect predictions')
    
    x_fit = np.array([axmin, axmax])
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, 'r', lw=2, label='Linear fit') # Fitted line
    
    # Regression Equation Text
    eq_label = f"y = {slope:.3f}x + {intercept:.2f}"
    ax.plot([], [], ' ', label=eq_label)  # ' ' makes the marker/line invisible

    if show_xaxis:
        ax.set_xlabel(r'Measured PM$_{2.5}$ ($\mu$g/m$^{3}$)', fontsize=fontsize)

    else:
        ax.set_xlabel('')
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        ax.tick_params(axis='y', labelsize=fontsize)

    if show_yaxis:
        ax.set_ylabel(r'Predicted PM$_{2.5}$ ($\mu$g/m$^{3}$)', fontsize=fontsize)

    else:
        ax.set_ylabel('')
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        ax.tick_params(axis='x', labelsize=fontsize)
        
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.set_ylim(axmin, axmax)
    ax.set_xlim(axmin, axmax)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, linestyle='dotted')
    ax.set_title(title, fontsize=fontsize)

def plot_scatter_density(ax, obs, pred, axmin=-20, axmax=400, fontsize=16, title='', show_xaxis=True, show_yaxis=True):
    # Ensure inputs are numpy arrays for calculation
    obs_arr = np.asarray(obs)
    pred_arr = np.asarray(pred)
    
    # 1. Calculate the point density
    xy = np.vstack([obs_arr, pred_arr])
    z = gaussian_kde(xy)(xy)
    
    # 2. Sort the points by density so that the densest points are plotted on top
    idx = z.argsort()
    obs_sorted = obs_arr[idx]
    pred_sorted = pred_arr[idx]
    z_sorted = z[idx]

    # Calculate regression (assuming fit_linear is defined elsewhere in your script)
    slope, intercept, residuals = fit_linear(obs_arr, pred_arr)
    
    # 3. Update scatter to use the density array 'z_sorted' as the color 'c' parameter
    # The 'jet' colormap closely resembles the blue-to-red transition in your image
    sc = ax.scatter(obs_sorted, pred_sorted, c=z_sorted, s=5, cmap='jet', label='Predictions')
    
    ax.plot([axmin, axmax],
            [axmin, axmax], 'k--', lw=1, label='Perfect predictions')
    
    x_fit = np.array([axmin, axmax])
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, 'r', lw=2, label='Linear fit') # Fitted line
    
    # Regression Equation Text
    eq_label = f"y = {slope:.3f}x + {intercept:.2f}"
    ax.plot([], [], ' ', label=eq_label)  # ' ' makes the marker/line invisible

    if show_xaxis:
        ax.set_xlabel(r'Measured PM$_{2.5}$ ($\mu$g/m$^{3}$)', fontsize=fontsize)
    else:
        ax.set_xlabel('')
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        ax.tick_params(axis='y', labelsize=fontsize)

    if show_yaxis:
        ax.set_ylabel(r'Predicted PM$_{2.5}$ ($\mu$g/m$^{3}$)', fontsize=fontsize)
    else:
        ax.set_ylabel('')
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        ax.tick_params(axis='x', labelsize=fontsize)
        
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.set_ylim(axmin, axmax)
    ax.set_xlim(axmin, axmax)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, linestyle='dotted')
    ax.set_title(title, fontsize=fontsize)

    return sc