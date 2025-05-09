import json
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def load_and_generate_text(
    model_path, 
    prompts=None, 
    max_length=10, 
    temperature=0.7, 
    top_p=0.9, 
    top_k=50
):
    """
    Load a fine-tuned model and generate text for given prompts
    
    Args:
        model_path (str): Path to the saved model
        prompts (list): List of prompts to generate text for
        max_length (int): Maximum length of generated text
        temperature (float): Sampling temperature for text generation
        top_p (float): Nucleus sampling parameter
        top_k (int): Top-k sampling parameter
    
    Returns:
        list: Generated texts for each prompt
    """
    try:        
        # Default prompts if none provided
        if prompts is None:
            prompts = [
                "Who is Leonardo Da Vinci?",
                "What is the meaning of life?",
                "Explain quantum physics in simple terms."
            ]
        
        # Create text generation pipeline
        pipe = pipeline(
            task="text-generation", 
            model=model_path, 
            tokenizer=model_path,
            max_length=max_length,
            temperature=temperature,
            device="cuda:0",
            truncation=True,
            top_p=top_p,
            top_k=top_k
        )
        
        # Generate text for each prompt
        results = []
        for prompt in prompts:
            full_prompt = f"<s>[INST] {prompt} [/INST]"
            result = pipe(full_prompt)
            
            # Extract and store generated text
            if result and len(result) > 0:
                generated_text = result[0]['generated_text']
                results.append({
                    'prompt': prompt,
                    'generated_text': generated_text
                })
            else:
                results.append({
                    'prompt': prompt,
                    'generated_text': "Failed to generate text."
                })
        
        return results
    
    except Exception as e:
        print(f"Error in text generation: {e}")
        return []


# Function to extract random inputs from a JSONL file
def extract_random_inputs(jsonl_file_path, num_samples):
    inputs = []
    pairs = []

    # Read the JSONL file and collect all "input" fields
    with open(jsonl_file_path, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            if "input" in data:
                pairs.append(data["input"])
                pairs.append(data["output"]["violation"])
                inputs.append(pairs)
                pairs = []


    # Select random samples
    if len(inputs) < num_samples:
        print(f"Warning: File contains fewer than {num_samples} entries. Returning all available inputs.")
        return inputs

    random_inputs = random.sample(inputs, num_samples)
    return random_inputs

def main():
    # Path to your fine-tuned model
    model_path = "./llama-2-7b-chat-violation-checker"
        

    # Specify the path to your JSONL file and number of samples
    jsonl_file_path = "support_converted.jsonl"  # Replace with your file path
    num_samples = 3

    # Extract random inputs
    random_inputs_list = extract_random_inputs(jsonl_file_path, num_samples)
    custom_prompts = []
    ground_truth = []
        
    for i in random_inputs_list:
        g = i[0]
        ground_truth.append(i[1])
        custom_prompts.append(f"If the given sentence ```{g}``` is posted in any social media, will it violate any rule? If violation is true then which rule is violated and what is the explanation. Your answer must strictly include if violation is true or false.")

    results = load_and_generate_text(
        model_path, 
        prompts=custom_prompts, 
        max_length=300  # Increased max length for more detailed responses
    )

    # # Print results
    d = 0
    for result in results:
        print("\n--- Prompt ---")
        print(result['prompt'])
        print("\n--- Generated Text ---")
        print(result['generated_text'])
        print("-" * 50)
        print(f"True value of violation  = {ground_truth[d]}")
        d=d+1
        


if __name__ == "__main__":
    main()