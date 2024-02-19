import os
import json
import torch
from src.utils.model_utils import load_gpt_model_and_tokenizer
from src.utils.prompt_utils import load_dataset, word_pairs_to_prompt_data, create_prompt
from src.utils.extract_utils import get_mean_head_activations, compute_universal_function_vector

def generate_function_vector_json(model_name, dataset_file_name, edit_layer = 9):
    """
    Generate function vector JSON file for a given GPT model and dataset.

    Args:
    - model_name: Name of the GPT model
    - dataset_file_name: Name of the dataset file
    - edit_layer: Layer of model to use for function vector computation

    Returns:
    - None
    """
    # Load GPT model and tokenizer
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name)
    EDIT_LAYER = edit_layer
    
    # Load dataset
    dataset = load_dataset(dataset_file_name, seed=0)
    
    # Compute mean activations
    mean_activations = get_mean_head_activations(dataset, model, model_config, tokenizer)
    
    # Compute function vector and top heads
    FV, top_heads = compute_universal_function_vector(mean_activations, model, model_config, n_top_heads=10)
    
    # Create data directory if not exists
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Save the function vector and top heads in a JSON file
    with open(f'data/{dataset_file_name}.json', 'w') as f:
        json.dump({'FV': FV.tolist(), 'mean_activations': mean_activations.tolist(), 'top_heads': top_heads}, f)
    
    print(f"Function vector JSON file saved as data/{dataset_file_name}.json")
    return

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate function vector JSON")
    parser.add_argument("--model_name", type=str, help="Name of the GPT model")
    parser.add_argument("--dataset_file_name", type=str, help="Name of the dataset file")
    parser.add_argument("--dataset_file_name", type=str, help="Layer of model", default=9)
    args = parser.parse_args()

    generate_function_vector_json(args.model_name, args.dataset_file_name)
