import sys
import os
import re
import yaml

def update_slurm_script(model_name, logs_dir, time, ngpus, mem):
    slurm_script_path = 'run.sh'  # Fixed path to the SLURM batch script

    model_name = model_name.split('/')[1]

    # Define patterns to identify the --out and --err file paths
    out_pattern = re.compile(r'#SBATCH\s+--out=(\S+)')
    err_pattern = re.compile(r'#SBATCH\s+--err=(\S+)')
    mem_pattern = re.compile(r'#SBATCH\s+--mem=(\S+)')
    time_pattern = re.compile(r'#SBATCH\s+--time (\S+)')
    gres_pattern = re.compile(r'#SBATCH\s+--gres=gpu:(\S+)')

    # Read the content of the original SLURM script
    with open(slurm_script_path, 'r') as file:
        script_content = file.read()

    # Find the current --out and --err file paths in the script
    current_out_match = out_pattern.search(script_content)
    current_err_match = err_pattern.search(script_content)
    current_mem_match = mem_pattern.search(script_content)
    current_time_match = time_pattern.search(script_content)
    current_gres_match = gres_pattern.search(script_content)

    if not current_out_match or not current_err_match or not current_mem_match or not current_time_match or not current_gres_match:
        print("Error: Unable to find all params in the SLURM script.")
        sys.exit(1)

    # Generate new output and error file names based on the model name
    new_out_path = f'{logs_dir}/{model_name}.out'
    new_err_path = f'{logs_dir}/{model_name}.err'

    # Replace the --out and --err paths with the new file names in the script content
    updated_content = gres_pattern.sub(f'#SBATCH --gres=gpu:{ngpus}', script_content)
    updated_content = mem_pattern.sub(f'#SBATCH --mem={mem}', updated_content)
    updated_content = time_pattern.sub(f'#SBATCH --time {time}', updated_content)
    updated_content = out_pattern.sub(f'#SBATCH --out={new_out_path}', updated_content)
    updated_content = err_pattern.sub(f'#SBATCH --err={new_err_path}', updated_content)

    # Write the updated script content back to the SLURM script
    with open(slurm_script_path, 'w') as file:
        file.write(updated_content)

    print(f"Updated SLURM script '{slurm_script_path}' with new output and error file names.")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python update_slurm_script.py <model_name>")
        sys.exit(1)
    with open("config.yaml", 'r') as file:
        params = yaml.safe_load(file)
    logs_dir = params["logs_dir"]
    time = params["time"]
    mem = params["mem"]
    ngpus = params["ngpus"]
    model_name = sys.argv[1]
    update_slurm_script(model_name, logs_dir, time=time, mem=mem, ngpus=ngpus)
