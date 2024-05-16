import sys
import os
import re

def update_slurm_script(model_name):
    slurm_script_path = 'run.sh'  # Fixed path to the SLURM batch script

    model_name = model_name.split('/')[1]

    # Define patterns to identify the --out and --err file paths
    out_pattern = re.compile(r'#SBATCH\s+--out=(\S+)')
    err_pattern = re.compile(r'#SBATCH\s+--err=(\S+)')

    # Read the content of the original SLURM script
    with open(slurm_script_path, 'r') as file:
        script_content = file.read()

    # Find the current --out and --err file paths in the script
    current_out_match = out_pattern.search(script_content)
    current_err_match = err_pattern.search(script_content)

    if not current_out_match or not current_err_match:
        print("Error: Unable to find --out or --err paths in the SLURM script.")
        sys.exit(1)

    current_out_path = current_out_match.group(1)
    current_err_path = current_err_match.group(1)

    # Generate new output and error file names based on the model name
    new_out_path = re.sub(r'([^/]+)\.out$', f'llm4se_inference_{model_name}.out', current_out_path)
    new_err_path = re.sub(r'([^/]+)\.err$', f'llm4se_inference_{model_name}.err', current_err_path)

    # Replace the --out and --err paths with the new file names in the script content
    updated_content = out_pattern.sub(f'#SBATCH --out={new_out_path}', script_content)
    updated_content = err_pattern.sub(f'#SBATCH --err={new_err_path}', updated_content)

    # Write the updated script content back to the SLURM script
    with open(slurm_script_path, 'w') as file:
        file.write(updated_content)

    print(f"Updated SLURM script '{slurm_script_path}' with new output and error file names.")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python update_slurm_script.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    update_slurm_script(model_name)

