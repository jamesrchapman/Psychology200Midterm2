import numpy as np
from scipy.stats import chi2


def main():
    observed = np.array([
            [24, 28, 45],
            [23, 61, 19]
        ])
    df = 2
    cross_tabulation = add_row_and_column_totals(observed)
    expected = calculate_expected_values(cross_tabulation)
    pearson_res = calculate_pearson_residuals(observed,expected)
    chi_square_contribution = np.square(pearson_res)
    chi_square = np.sum(chi_square_contribution)
    p_value = 1 - chi2.cdf(chi_square, df)
    standardized_res = calculate_standardized_residuals(cross_tabulation,observed,expected)
    row_percentages = calculate_row_percentages(observed)
    expected_row_percentages = calculate_row_percentages(expected)

    print("Observed:\n", observed)
    print("Cross Tabulation:\n", cross_tabulation)
    print("Expectation:\n", expected)
    print("Pearson Residuals:\n", pearson_res)
    print("Contribution to Chi-Square", chi_square_contribution)
    print("Chi-Square", chi_square)
    print("p-value",p_value)
    print("Standardized Pearson Residuals:\n", standardized_res)
    print("Observed Row Percentages:\n", row_percentages)
    print("Expected Row Percentages:\n", expected_row_percentages)


def add_row_and_column_totals(matrix):    
    # Calculate row sums and add as a new column
    row_sums = np.sum(matrix, axis=1).reshape(-1, 1)
    matrix_with_row_totals = np.hstack((matrix, row_sums))

    # Calculate column sums and add as a new row (including row totals)
    column_sums = np.sum(matrix_with_row_totals, axis=0).reshape(1, -1)
    final_matrix = np.vstack((matrix_with_row_totals, column_sums))

    # Display result
    # print("Raw Matrix:\n", matrix)
    # print("\nObservation Matrix with Row and Column Totals:\n", final_matrix)
    return final_matrix

def calculate_expected_values(matrix):
    # Separate the original matrix (exclude the last row and column)
    original_matrix = matrix[:-1, :-1]
    
    # Extract row and column totals (last column and row, excluding the grand total)
    row_totals = matrix[:-1, -1]
    # print("Row Totals:\n", row_totals)
    column_totals = matrix[-1, :-1]
    # print("Column Totals:\n", column_totals)

    # Grand total (bottom-right corner of the matrix)
    grand_total = matrix[-1, -1]
    # print("Sample Size:\n", grand_total)

    # Calculate expected values using (row_total * column_total) / grand_total
    # print("Row*Col Outer Product:\n", np.outer(row_totals,column_totals))
    expected_values = np.outer(row_totals, column_totals) / grand_total
    # print("Expectation Matrix:\n", expected_values)
    return expected_values

def calculate_pearson_residuals(observed, expected):
    # Calculate the Pearson residuals matrix
    residuals = (observed - expected) / np.sqrt(expected)
    return residuals

def calculate_standardized_residuals(matrix_with_totals, observed, expected):
    # Extract row totals and column totals
    row_totals = matrix_with_totals[:-1, -1]
    column_totals = matrix_with_totals[-1, :-1]
    grand_total = matrix_with_totals[-1, -1]

    # Calculate the standardized residuals matrix
    residuals = (observed - expected) / np.sqrt(
        expected * (1 - row_totals[:, np.newaxis] / grand_total) * (1 - column_totals / grand_total)
    )
    return residuals

def calculate_row_percentages(observed):
    # Calculate row totals
    row_totals = np.sum(observed, axis=1).reshape(-1, 1)
    
    # Calculate percentage of each element relative to its row total
    row_percentages = (observed / row_totals) * 100
    
    return row_percentages


if __name__ == '__main__':
    main()

"""
Output:

Observed:
 [[24 28 45]
 [23 61 19]]
Cross Tabulation:
 [[ 24  28  45  97]
 [ 23  61  19 103]
 [ 47  89  64 200]]
Expectation:
 [[22.795 43.165 31.04 ]
 [24.205 45.835 32.96 ]]
Pearson Residuals:
 [[ 0.25238716 -2.3082165   2.50567397]
 [-0.24492578  2.23997823 -2.43159822]]
Contribution to Chi-Square [[0.06369928 5.32786343 6.27840206]
 [0.05998864 5.01750245 5.9126699 ]]
Chi-Square 22.660125765113058
p-value 1.2006494063965256e-05
Standardized Pearson Residuals:
 [[ 0.40209893 -4.31744124  4.23415245]
 [-0.40209893  4.31744124 -4.23415245]]
Observed Row Percentages:
 [[24.74226804 28.86597938 46.39175258]
 [22.33009709 59.22330097 18.44660194]]
"""