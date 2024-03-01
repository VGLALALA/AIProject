import numpy as np

# Creating an array
a = np.array([1, 2, 3, 4, 5, 6])

# Reshaping the array into a 2x3 matrix
a_reshaped = a.reshape((2, 3))

# Generating a range of numbers with arange and reshaping
b = np.arange(1, 13).reshape((3, 4))

# Matrix multiplication
product = np.dot(a_reshaped, b)

# Element-wise multiplication
element_wise_product = a_reshaped * np.array([[2], [3]])

# Summation of all elements in the matrix
sum_of_elements = np.sum(b)

# Computing the mean of each column
mean_of_columns = np.mean(b, axis=0)

# Finding the maximum element of each row
max_of_rows = np.max(b, axis=1)

# Creating a boolean mask where elements of b are greater than 5
mask = b > 5

# Applying the mask to b
filtered_b = b[mask]

# Concatenating a_reshaped with itself horizontally
concatenated_horizontally = np.hstack((a_reshaped, a_reshaped))

# Creating an identity matrix
identity_matrix = np.eye(3)

# Dictionary of variables and their names
variables = {
    "a_reshaped": a_reshaped,
    "b": b,
    "product": product,
    "element_wise_product": element_wise_product,
    "sum_of_elements": sum_of_elements,
    "mean_of_columns": mean_of_columns,
    "max_of_rows": max_of_rows,
    "filtered_b": filtered_b,
    "concatenated_horizontally": concatenated_horizontally,
    "identity_matrix": identity_matrix
}

# Printing variable names with their output
for name, value in variables.items():
    print(f"{name}: \n{value}\n")
