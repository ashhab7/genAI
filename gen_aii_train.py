import json
import sys
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    logging
)
from peft import LoraConfig
from trl import SFTTrainer
from datasets import Dataset

def verify_jsonl_format(input_file):
    """
    Verifies that each line in the input JSONL file follows the expected format.
    
    Expected format:
    {
        "input": "text string",
        "output": {
            "violation": boolean,
            "rule": "string",
            "explanation": "string"
        },
        "id": integer
    }
    
    Returns a tuple of (is_valid, errors, data) where:
    - is_valid: boolean indicating if the file is valid
    - errors: list of error messages (empty if valid)
    - data: list of parsed JSON objects from the file (empty if there were errors)
    """
    line_number = 0
    errors = []
    data = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line_number += 1
                line = line.strip()
                
                if not line:  # Skip empty lines
                    continue
                
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    errors.append(f"Line {line_number}: Invalid JSON format")
                    continue
                
                # Check for required fields
                if "input" not in item:
                    errors.append(f"Line {line_number}: Missing 'input' field")
                elif not isinstance(item["input"], str) and not isinstance(item["input"], dict):
                    errors.append(f"Line {line_number}: 'input' field must be a string or dict")
                
                if "output" not in item:
                    errors.append(f"Line {line_number}: Missing 'output' field")
                elif not isinstance(item["output"], dict):
                    errors.append(f"Line {line_number}: 'output' field must be a dictionary")
                else:
                    # Check output structure
                    output = item["output"]
                    
                    if "violation" not in output:
                        errors.append(f"Line {line_number}: Missing 'violation' field in output")
                    elif not isinstance(output["violation"], bool):
                        errors.append(f"Line {line_number}: 'violation' field must be a boolean")
                    
                    if "rule" not in output:
                        errors.append(f"Line {line_number}: Missing 'rule' field in output")
                    elif not isinstance(output["rule"], str):
                        errors.append(f"Line {line_number}: 'rule' field must be a string")
                    
                    if "explanation" not in output:
                        errors.append(f"Line {line_number}: Missing 'explanation' field in output")
                    elif not isinstance(output["explanation"], str):
                        errors.append(f"Line {line_number}: 'explanation' field must be a string")
                
                if "id" not in item:
                    errors.append(f"Line {line_number}: Missing 'id' field")
                elif not isinstance(item["id"], int):
                    errors.append(f"Line {line_number}: 'id' field must be an integer")
                
                # If this item passed all checks, add it to our data
                if not errors or errors[-1].split(':')[0] != f"Line {line_number}":
                    data.append(item)
    
    except FileNotFoundError:
        return False, [f"File '{input_file}' not found"], []
    except Exception as e:
        return False, [f"Error reading file: {str(e)}"], []
    
    return len(errors) == 0, errors, data

def preprocess_data(data):
    """
    Formats the data for training in the format required by SFTTrainer.
    """
    processed_texts = []
    
    for item in data:
        input_post = item['input']['post'] if isinstance(item['input'], dict) else item['input']
        
        # Create a formatted text string suitable for instruction tuning
        formatted_text = (
            f"### Instruction:\n"
            f"For the given sentence, determine if it violates any rule. If it does, identify which rule is violated and explain why.\n\n"
            f"### Input:\n"
            f"{input_post}\n\n"
            f"### Response:\n"
            f"Violation: {item['output']['violation']}\n"
            f"Rule: {item['output']['rule']}\n"
            f"Explanation: {item['output']['explanation']}"
        )
        
        processed_texts.append(formatted_text)
    
    return Dataset.from_dict({"text": processed_texts})

def main():
    # Set GPU device if needed
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Configuration 
    input_file = 'support_converted.jsonl'
    output_dir = './results'
    base_model = "NousResearch/Llama-3.2-1B"  # Using chat model for better instruction following
    new_model = "llama-2-7b-chat-violation-checker-light"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    
    print(f"Verifying JSONL format of '{input_file}'...")
    is_valid, errors, json_data = verify_jsonl_format(input_file)
    
    if not is_valid:
        print(f"Found {len(errors)} errors in the JSONL file:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease fix these errors before proceeding with training.")
        sys.exit(1)
    
    # If we're here, the file is valid
    print(f"Validation successful! All records in '{input_file}' follow the expected format.")
    
    # Print some statistics
    violation_count = sum(1 for record in json_data if record["output"]["violation"])
    
    print(f"Total records: {len(json_data)}")
    print(f"Records with violations: {violation_count} ({violation_count/len(json_data)*100:.2f}%)")
    print(f"Records without violations: {len(json_data) - violation_count} ({(len(json_data) - violation_count)/len(json_data)*100:.2f}%)")
    
    # Preprocess the data for SFTTrainer
    print("Preprocessing data...")
    dataset = preprocess_data(json_data)
    print(f"Processed {len(dataset)} records for training")
    
    try:
        # Load tokenizer and model
        print(f"Loading model: {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            torch_dtype=torch.float16,
            token="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"
        )
        
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        # LoRA configuration
        print("Setting up LoRA configuration...")
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Training arguments
        print("Setting up training arguments...")
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            learning_rate=2e-4,
            weight_decay=0.001,
            fp16=torch.cuda.is_available(),
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            logging_steps=10,
            save_strategy="epoch",
            remove_unused_columns=False,
        )
        
        # Initialize trainer
        print("Setting up SFTTrainer...")
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_config,
            args=training_args,
        )
        
        # Train the model
        print("Starting training...")
        trainer.train()
        
        # Save the model
        print(f"Saving fine-tuned model to {new_model}...")
        trainer.model.save_pretrained(new_model)
        tokenizer.save_pretrained(new_model)
        
        print("Training complete!")
        
    except Exception as e:
        print(f"An error occurred during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()