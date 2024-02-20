import os
import json
import torch
from src.utils.model_utils import load_gpt_model_and_tokenizer
from src.utils.prompt_utils import load_dataset, word_pairs_to_prompt_data, create_prompt
from src.utils.extract_utils import get_mean_head_activations, compute_universal_function_vector

def generate_function_vector_json(model, tokenizer, model_config, dataset_file_name, edit_layer=9):
    """
    Generate function vector JSON file for a given GPT model and dataset.

    Args:
    - model: The GPT model.
    - tokenizer: The tokenizer for the GPT model.
    - model_config: The configuration of the GPT model.
    - dataset_file_name: Name of the dataset file.
    - edit_layer: Layer of model to use for function vector computation.

    Returns:
    - None
    """
    EDIT_LAYER = edit_layer
    
    # Load dataset
    dataset = load_dataset(dataset_file_name, seed=0)
    
    # Compute mean activations
    mean_activations = get_mean_head_activations(dataset, model, model_config, tokenizer)
    
    # Compute function vector and top heads
    FV, top_heads = compute_universal_function_vector(mean_activations, model, model_config, n_top_heads=10)
    
    # Create data directory if not exists
    if not os.path.exists('../data'):
        os.makedirs('../data')
    
    # Define the path to the JSON file
    json_file_path = '../data/function_vectors.json'
    
    # Check if the JSON file already exists
    if os.path.exists(json_file_path):
        # If the file exists, load its contents
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    else:
        # If the file doesn't exist, initialize an empty dictionary
        data = {}
    
    # Add or update the dataset_file_name entry in the dictionary
    data[dataset_file_name] = {'FV': FV.tolist()}
    
    # Save the updated dictionary to the JSON file
    with open(json_file_path, 'w') as f:
        json.dump(data, f)
    
    print(f"Function vector JSON data appended to {json_file_path}")
    return



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate function vector JSON")
    parser.add_argument("--model_name", type=str, help="Name of the GPT model")
    parser.add_argument("--dataset_file_name", type=str, help="Name of the dataset file")
    parser.add_argument("--dataset_file_name", type=str, help="Layer of model", default=9)
    args = parser.parse_args()

    generate_function_vector_json(args.model_name, args.dataset_file_name)
