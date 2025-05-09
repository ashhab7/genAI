import json
import random
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def load_and_generate_text(
    model_path, 
    prompts=None, 
    max_length=300, 
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
        
        print(f"Loading model from {model_path}...")
        # Create text generation pipeline
        pipe = pipeline(
            task="text-generation", 
            model=model_path, 
            tokenizer=model_path,
            max_length=max_length,
            temperature=temperature,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            truncation=True,
            top_p=top_p,
            top_k=top_k
        )
        
        print(f"Generating responses for {len(prompts)} prompts...")
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

def read_sentences_from_file(file_path):
    """
    Reads sentences from a text file, one sentence per line
    
    Args:
        file_path (str): Path to the text file
        
    Returns:
        list: List of sentences
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            sentences = [line.strip() for line in file if line.strip()]
        return sentences
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []

def format_prompts(sentences):
    """
    Formats sentences into custom prompts for rule violation checking
    
    Args:
        sentences (list): List of sentences to format
        
    Returns:
        list: Formatted prompts
    """
    custom_prompts = []
    for sentence in sentences:
        prompt = (f"If the given sentence ```{sentence}``` is posted in any social media, "
                 f"will it violate any rule? If violation is true then which rule is violated "
                 f"and what is the explanation. Your answer must strictly include if violation is true or false.")
        custom_prompts.append(prompt)
    return custom_prompts

def save_results_to_file(results, output_file):
    """
    Saves the generated results to a file
    
    Args:
        results (list): List of dictionaries containing prompts and generated text
        output_file (str): Path to the output file
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            for i, result in enumerate(results, 1):
                file.write(f"=== Sample {i} ===\n\n")
                file.write("--- Original Text ---\n")
                # Extract the original text from the prompt
                original_text = result['prompt'].split('```')[1].split('```')[0]
                file.write(f"{original_text}\n\n")
                file.write("--- Model Response ---\n")
                # Extract just the response part (after [/INST])
                if "[/INST]" in result['generated_text']:
                    response = result['generated_text'].split("[/INST]")[1].strip()
                else:
                    response = result['generated_text']
                file.write(f"{response}\n\n")
                file.write("-" * 50 + "\n\n")
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results to {output_file}: {e}")

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Generate text from a fine-tuned LLaMA model using sentences from a text file.')
    parser.add_argument('--model', type=str, default="./llama-2-7b-chat-violation-checker-light", help='Path to the fine-tuned model')
    parser.add_argument('--input', type=str, help='Path to the input text file with sentences', default='input.txt')
    parser.add_argument('--output', type=str, default='model_responses.txt', help='Path to save the generated responses')
    parser.add_argument('--max_length', type=int, default=300, help='Maximum length for generated responses')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for text generation')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p value for nucleus sampling')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k value for sampling')
    
    args = parser.parse_args()
    
    # Read sentences from the input file
    print(f"Reading sentences from {args.input}...")
    sentences = read_sentences_from_file(args.input)
    
    if not sentences:
        print("No sentences found in the input file. Exiting.")
        return
    
    print(f"Found {len(sentences)} sentences in the input file.")
    
    # Format sentences into prompts
    custom_prompts = format_prompts(sentences)
    
    # Generate text using the model
    results = load_and_generate_text(
        model_path=args.model, 
        prompts=custom_prompts, 
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k
    )
    
    # Print results to console
    for i, result in enumerate(results, 1):
        print(f"\n=== Sample {i} ===")
        original_text = result['prompt'].split('```')[1].split('```')[0]
        print(f"Original Text: {original_text}")
        print("Model Response:")
        if "[/INST]" in result['generated_text']:
            response = result['generated_text'].split("[/INST]")[1].strip()
            print(response)
        else:
            print(result['generated_text'])
        print("-" * 50)
    
    # Save results to file
    save_results_to_file(results, args.output)

if __name__ == "__main__":
    main()