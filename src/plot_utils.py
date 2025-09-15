import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

def ensure_out():
    os.makedirs('out', exist_ok=True)

def plot_regression_line(X, y, y_pred, title='Linear Regression (Best Fit Line)', filename='out/linear_fit.png'):
    ensure_out()
    X = np.asarray(X).ravel()
    plt.figure()
    plt.scatter(X, y, label='Actual')
    plt.plot(X, y_pred, linewidth=2, label='Prediction')
    plt.title(title)
    plt.xlabel('Feature (e.g., study_hours)')
    plt.ylabel('Final Mark')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_polynomial_curve(X, y, y_pred, title='Polynomial Regression (Curve Fit)', filename='out/polynomial_fit.png'):
    ensure_out()
    order = np.argsort(X.ravel())
    Xs, ys, yp = X.ravel()[order], np.asarray(y)[order], np.asarray(y_pred)[order]
    plt.figure()
    plt.scatter(Xs, ys, label='Actual')
    plt.plot(Xs, yp, linewidth=2, label='Prediction')
    plt.title(title)
    plt.xlabel('Feature (e.g., study_hours)')
    plt.ylabel('Final Mark')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_logistic_prob(X, y_binary, p1, title='Logistic Regression (Pass Probability)', filename='out/logistic_prob.png'):
    ensure_out()
    order = np.argsort(X.ravel())
    Xs, yb, p = X.ravel()[order], np.asarray(y_binary)[order], np.asarray(p1)[order]
    plt.figure()
    plt.scatter(Xs, yb, label='Actual Label (0/1)')
    plt.plot(Xs, p, linewidth=2, label='P(pass)')
    plt.title(title)
    plt.xlabel('Feature (e.g., assign_avg)')
    plt.ylabel('Probability / Label')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
