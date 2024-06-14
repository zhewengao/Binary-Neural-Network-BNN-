def read_weights_matrices(filename):
    """
    Read weights matrices from a file.

    Args:
        filename (str): The path to the file containing the weights matrices.

    Returns:
        list: A list of tuples, where each tuple contains the name of the matrix and its corresponding values.

    Example:
        >>> matrices = read_weights_matrices('weights.txt')
        >>> print(matrices)
        [('fc1', [[1, 0, 1], [0, 1, 0], [1, 0, 1]]), ('fc2', [[1, 1, 1], [0, 0, 0], [1, 1, 1]])]
    """
    matrices = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        matrix_name = None
        matrix_values = []
        for line in lines:
            if line.startswith("fc"):
                # New matrix name found, save the previous matrix
                if matrix_name and matrix_values:
                    matrices.append((matrix_name, matrix_values))
                matrix_name = line.strip()
                matrix_values = []
            else:
                # Read matrix values (convert to 0 or 1)
                row_values = [int(float(val)) for val in line.split()]
                matrix_values.append(row_values)
        # Save the last matrix
        if matrix_name and matrix_values:
            matrices.append((matrix_name, matrix_values))
    return matrices

def convert_to_2d_array(matrix_values):
    return [['1' if val == 1 else '0' for val in row] for row in matrix_values]

def write_to_text_file(matrices, output_filename):
    with open(output_filename, 'w') as output_file:
        for matrix_name, matrix_values in matrices:
            output_file.write(f"Matrix {matrix_name}:\n")
            for row in convert_to_2d_array(matrix_values):
                output_file.write("{" + ", ".join(row) + "},\n")

def main():
    """
    Converts weight matrices from an input file to a text file.

    Reads weight matrices from the input file specified by `input_filename`,
    converts them, and writes the converted matrices to the output file
    specified by `output_filename`. Prints a message indicating the location
    where the converted matrices are saved.

    Args:
        None

    Returns:
        None
    """
    input_filename = 'model.txt'  # Replace with your actual input file name
    output_filename = 'converted_weights.txt'  # Replace with your desired output file name
    matrices = read_weights_matrices(input_filename)
    write_to_text_file(matrices, output_filename)
    print(f"Converted matrices saved to {output_filename}")

if __name__ == "__main__":
    main()