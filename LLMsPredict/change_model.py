import sys
import yaml

def update_model_name(new_model_name):
    config_path = 'config.yaml'

    # Load YAML configuration file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Update the model_name with the new_model_name
    config['model_name'] = new_model_name

    # Save the updated configuration back to the file
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

    print(f"Model name in {config_path} updated to: {new_model_name}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python update_model_name.py <new_model_name>")
        sys.exit(1)

    new_model_name = sys.argv[1]
    update_model_name(new_model_name)

