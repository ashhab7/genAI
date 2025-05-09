import json
import random
import torch
import argparse
import re
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score
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

def read_data_with_labels(file_path):
    """
    Reads sentences with their ground truth labels from a file.
    Each line should be formatted as: "sentence|label" where label is true/false or 1/0
    
    Args:
        file_path (str): Path to the text file
        
    Returns:
        tuple: (sentences, labels)
    """
    sentences = []
    labels = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                    
                # Check for different separators (| or ,)
                if '|' in line:
                    parts = line.split('|', 1)
                elif ',' in line:
                    parts = line.split(',', 1)
                else:
                    # If no separator, assume it's just a sentence without a label
                    parts = [line, None]
                
                # Extract sentence and label
                if len(parts) >= 2:
                    sentence = parts[0].strip()
                    label_str = parts[1].strip().lower()
                    
                    # Convert label string to boolean
                    if label_str in ['true', '1', 'yes', 't', 'y']:
                        label = True
                    elif label_str in ['false', '0', 'no', 'f', 'n']:
                        label = False
                    else:
                        print(f"Warning: Unrecognized label '{label_str}' for sentence '{sentence}'. Skipping.")
                        continue
                    
                    sentences.append(sentence)
                    labels.append(label)
                else:
                    print(f"Warning: No label found for line '{line}'. Skipping.")
        
        return sentences, labels
    except Exception as e:
        print(f"Error reading labeled data from {file_path}: {e}")
        return [], []

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

def parse_model_prediction(generated_text):
    """
    Parse the model's generated text to extract whether it predicted a violation.
    
    Args:
        generated_text (str): The text generated by the model
        
    Returns:
        bool: True if the model predicted a violation, False otherwise
    """
    # Extract just the response part (after [/INST])
    if "[/INST]" in generated_text:
        text = generated_text.split("[/INST]")[1].strip()
    else:
        text = generated_text
    
    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Look for explicit statements about violations
    if re.search(r'violation\s+is\s+true', text_lower) or re.search(r'violation:\s*true', text_lower):
        return True
    elif re.search(r'violation\s+is\s+false', text_lower) or re.search(r'violation:\s*false', text_lower):
        return False
    
    # More flexible pattern matching
    violation_patterns = [
        r"violat(es|ion|ing)",
        r"break(s|ing)?\s+(the\s+)?rule",
        r"against\s+(the\s+)?rule",
        r"not\s+allowed",
        r"prohibited",
        r"content\s+policy\s+violat"
    ]
    
    non_violation_patterns = [
        r"no\s+violation",
        r"does\s+not\s+violate",
        r"doesn't\s+violate",
        r"not\s+a\s+violation",
        r"would\s+not\s+violate"
    ]
    
    # Check for non-violation patterns first (they're more specific)
    for pattern in non_violation_patterns:
        if re.search(pattern, text_lower):
            return False
    
    # Then check for violation patterns
    for pattern in violation_patterns:
        if re.search(pattern, text_lower):
            return True
    
    # Default to assuming no violation if unclear
    print(f"Warning: Could not clearly determine prediction from text. Defaulting to False.")
    print(f"Text excerpt: {text[:100]}...")
    return False

def calculate_metrics(y_true, y_pred):
    """
    Calculate classification metrics from ground truth and predictions.
    
    Args:
        y_true (list): Ground truth values (True/False)
        y_pred (list): Predicted values (True/False)
        
    Returns:
        dict: Dictionary containing various metrics
    """
    if not y_true:
        print("No ground truth labels provided. Skipping metrics calculation.")
        return None
    
    # Convert boolean lists to numpy arrays of 0 and 1
    y_true_np = np.array(y_true, dtype=int)
    y_pred_np = np.array(y_pred, dtype=int)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true_np, y_pred_np)
    
    # Extract values from confusion matrix
    if len(cm) == 2:  # If we have both positive and negative predictions
        tn, fp, fn, tp = cm.ravel()
    else:  # Handle the case where there's only one class in predictions
        if y_true[0] == True:  # If ground truth is all positives
            tp = sum(y_pred)
            fn = len(y_true) - tp
            tn, fp = 0, 0
        else:  # If ground truth is all negatives
            tn = sum([not p for p in y_pred])
            fp = len(y_true) - tn
            tp, fn = 0, 0
    
    # Calculate metrics
    accuracy = accuracy_score(y_true_np, y_pred_np)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_np, y_pred_np, average='binary', zero_division=0
    )
    
    # Create detailed report
    class_report = classification_report(
        y_true_np, y_pred_np, 
        target_names=["No Violation", "Violation"],
        zero_division=0,
        output_dict=True
    )
    
    return {
        "confusion_matrix": cm,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "classification_report": class_report
    }

def save_results_to_file(results, ground_truth, predictions, metrics, output_file):
    """
    Saves the generated results to a file, including predictions and metrics
    
    Args:
        results (list): List of dictionaries containing prompts and generated text
        ground_truth (list): List of ground truth labels (True/False)
        predictions (list): List of predicted labels (True/False)
        metrics (dict): Dictionary of metrics from calculate_metrics()
        output_file (str): Path to the output file
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            # Write individual samples
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
                
                # Add prediction and ground truth if available
                if predictions:
                    file.write(f"Predicted: {'Violation' if predictions[i-1] else 'No Violation'}\n")
                if ground_truth and i-1 < len(ground_truth):
                    file.write(f"Ground Truth: {'Violation' if ground_truth[i-1] else 'No Violation'}\n")
                    if predictions and i-1 < len(predictions):
                        file.write(f"Match: {'Yes' if ground_truth[i-1] == predictions[i-1] else 'No'}\n")
                
                file.write("-" * 50 + "\n\n")
            
            # Write metrics if available
            if metrics:
                file.write("\n" + "=" * 20 + " EVALUATION METRICS " + "=" * 20 + "\n\n")
                
                file.write("--- Confusion Matrix ---\n")
                file.write(f"True Positives (TP): {metrics['true_positives']}\n")
                file.write(f"True Negatives (TN): {metrics['true_negatives']}\n")
                file.write(f"False Positives (FP): {metrics['false_positives']}\n")
                file.write(f"False Negatives (FN): {metrics['false_negatives']}\n\n")
                
                file.write("--- Performance Metrics ---\n")
                file.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
                file.write(f"Precision: {metrics['precision']:.4f}\n")
                file.write(f"Recall: {metrics['recall']:.4f}\n")
                file.write(f"F1 Score: {metrics['f1_score']:.4f}\n\n")
                
                file.write("--- Classification Report ---\n")
                report = metrics["classification_report"]
                for class_name in ["No Violation", "Violation"]:
                    if class_name in report:
                        file.write(f"{class_name}:\n")
                        file.write(f"  Precision: {report[class_name]['precision']:.4f}\n")
                        file.write(f"  Recall: {report[class_name]['recall']:.4f}\n")
                        file.write(f"  F1-score: {report[class_name]['f1-score']:.4f}\n")
                        file.write(f"  Support: {report[class_name]['support']}\n\n")
                
                if "accuracy" in report:
                    file.write(f"Overall Accuracy: {report['accuracy']:.4f}\n")
                if "macro avg" in report:
                    file.write(f"Macro Avg F1: {report['macro avg']['f1-score']:.4f}\n")
                if "weighted avg" in report:
                    file.write(f"Weighted Avg F1: {report['weighted avg']['f1-score']:.4f}\n")
        
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results to {output_file}: {e}")

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Generate text from a fine-tuned LLaMA model using sentences from a text file.')
    parser.add_argument('--model', type=str, default="./llama-2-7b-chat-violation-checker", help='Path to the fine-tuned model')
    parser.add_argument('--input', type=str, required=True, help='Path to the input text file with sentences')
    parser.add_argument('--output', type=str, default='model_responses.txt', help='Path to save the generated responses')
    parser.add_argument('--max_length', type=int, default=300, help='Maximum length for generated responses')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for text generation')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p value for nucleus sampling')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k value for sampling')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model performance using ground truth labels')
    
    args = parser.parse_args()
    
    # Try to read sentences with labels first (for evaluation)
    sentences = []
    ground_truth = []
    
    if args.evaluate:
        print(f"Reading labeled data from {args.input}...")
        sentences, ground_truth = read_data_with_labels(args.input)
        if not sentences:
            print("No labeled data found. Falling back to reading sentences without labels.")
            args.evaluate = False
    
    # If not in evaluation mode, or if no labeled data was found, just read sentences
    if not args.evaluate:
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
    
    # Extract predictions from generated text
    predictions = []
    for result in results:
        prediction = parse_model_prediction(result['generated_text'])
        predictions.append(prediction)
    
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
        
        # Print prediction
        print(f"Predicted: {'Violation' if predictions[i-1] else 'No Violation'}")
        
        # Print ground truth if available
        if ground_truth and i-1 < len(ground_truth):
            print(f"Ground Truth: {'Violation' if ground_truth[i-1] else 'No Violation'}")
            print(f"Match: {'Yes' if ground_truth[i-1] == predictions[i-1] else 'No'}")
        
        print("-" * 50)
    
    # Calculate metrics if ground truth is available
    metrics = None
    if ground_truth:
        print("\nCalculating evaluation metrics...")
        metrics = calculate_metrics(ground_truth, predictions)
        
        if metrics:
            print("\n--- Confusion Matrix ---")
            print(f"True Positives (TP): {metrics['true_positives']}")
            print(f"True Negatives (TN): {metrics['true_negatives']}")
            print(f"False Positives (FP): {metrics['false_positives']}")
            print(f"False Negatives (FN): {metrics['false_negatives']}")
            
            print("\n--- Performance Metrics ---")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1_score']:.4f}")
            
            print("\n--- Classification Report ---")
            try:
                cr = classification_report(ground_truth, predictions, target_names=["No Violation", "Violation"])
                print(cr)
            except Exception as e:
                print(f"Could not generate classification report: {e}")
    
    # Save results to file
    save_results_to_file(results, ground_truth, predictions, metrics, args.output)
    print("Evaluation complete!")

if __name__ == "__main__":
    main()
