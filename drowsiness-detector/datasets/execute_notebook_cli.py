import argparse
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat import read, write

def execute_notebook(input_file):
    # Read the notebook
    with open(input_file, 'r') as f:
        nb = read(f, as_version=4)

    # Execute the notebook
    try:
        executor = ExecutePreprocessor(timeout=-1)
        executor.preprocess(nb, {'metadata': {'path': './'}})
    except Exception as e:
        raise RuntimeError(f"Notebook execution failed: {e}")

    # Save the executed notebook
    with open(input_file, 'w', encoding='utf-8') as f:
        write(nb, f)

    print("Notebook execution completed successfully.")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Execute the steps in a Jupyter Notebook.')
    parser.add_argument('input_file', type=str, help='Path to the input Jupyter Notebook file (.ipynb)')
    args = parser.parse_args()

    # Execute the notebook
    execute_notebook(args.input_file)
