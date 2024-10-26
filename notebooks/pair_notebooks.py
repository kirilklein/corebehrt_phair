import os
import subprocess

# Specify the path to your notebooks folder
notebooks_folder = os.getcwd()

# Iterate through all files in the notebooks folder
for filename in os.listdir(notebooks_folder):
    if filename.endswith(".ipynb"):
        # Construct the full path to the notebook
        notebook_path = os.path.join(notebooks_folder, filename)
        output_path = os.path.join(notebooks_folder, "scripts", filename.replace(".ipynb", ".py"))
        # Construct the jupytext command
        command = [
            "jupytext",
            "--set-formats", "ipynb,py:percent",
            notebook_path,
            "-o", output_path,
            "--pipe", "black",
        ]
        
        # Run the command
        try:
            subprocess.run(command, check=True)
            print(f"Successfully processed {filename}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {filename}: {e}")

print("All notebooks processed.")