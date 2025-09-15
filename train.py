from __future__ import annotations
import argparse
import numpy as np

from src.data_loader import load_dataset
from src.models import LinearRegressor, PolynomialRegressor, LogisticRegressor
from src.metrics import mse
from src.plot_utils import plot_regression_line, plot_polynomial_curve, plot_logistic_prob

def main():
    parser = argparse.ArgumentParser(description='Lesson 7: Regression & Logistic in OOP (Codespaces-safe).')
    parser.add_argument('--csv', type=str, default=None, help='Path to CSV (defaults to data/sample_marks.csv)')
    parser.add_argument('--poly_degree', type=int, default=3, help='Degree for polynomial regression.')
    parser.add_argument('--logit_threshold', type=float, default=0.5, help='Classification threshold for logistic.')
    args = parser.parse_args()

    df = load_dataset(args.csv)

    X_reg = df[['study_hours']].to_numpy()
    y_reg = df['final_mark'].to_numpy()

    X_log = df[['assign_avg']].to_numpy()
    y_log = df['passed'].to_numpy()

    lin = LinearRegressor().fit(X_reg, y_reg)
    y_lin = lin.predict(X_reg)
    print(f'[Linear] MSE: {mse(y_reg, y_lin):.3f}')
    plot_regression_line(X_reg, y_reg, y_lin, filename='out/linear_fit.png')

    poly = PolynomialRegressor(degree=args.poly_degree).fit(X_reg, y_reg)
    y_poly = poly.predict(X_reg)
    print(f'[Polynomial d={args.poly_degree}] MSE: {mse(y_reg, y_poly):.3f}')
    plot_polynomial_curve(X_reg, y_reg, y_poly, filename='out/polynomial_fit.png')

    logit = LogisticRegressor().fit(X_log, y_log)
    p_pass = logit.predict_proba(X_log)
    y_hat = logit.predict_label(X_log, threshold=args.logit_threshold)
    acc = float((y_hat == y_log).mean())
    print(f'[Logistic] Accuracy @ {args.logit_threshold:.2f}: {acc:.2%}')
    plot_logistic_prob(X_log, y_log, p_pass, filename='out/logistic_prob.png')

    print('Artifacts written to ./out')

if __name__ == '__main__':
    main()
