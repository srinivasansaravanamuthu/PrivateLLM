import os
import time
from flask import Flask, render_template, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Initialize Flask app
app = Flask(__name__)

# Load the model and tokenizer once
model_name = "EleutherAI/gpt-neo-2.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add pad_token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

# Print to verify the pad_token has been set
print("Pad token:", tokenizer.pad_token)

model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))  # Resize embeddings to include the new pad_token

# Set device to -1 to ensure it runs on CPU
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
print("Model EleutherAI/gpt-neo-2.7B loaded. Tokenizer and Pipeline created.")

def format_prompt(user_text):
    """
    Format the user input to provide more context for the model.
    """
    return f"Please provide a professional and factual answer without adding personal opinions to the following question:\nQ: {user_text}\nA:"

def post_process_response(response):
    """
    Post-process the model's response to ensure it is concise and appropriate.
    """
    # Ensure the response ends conclusively and truncate to 200 characters
    if "A:" in response:
        response = response.split("A:")[1].strip()
    
    response = response.split("Q:")[0].strip()  # Remove any additional questions
    response = response.split(". ")[0] + '.'  # Take only the first sentence
    return response[:200]  # Truncate to 200 characters
    #return response




@app.route('/')
def home():
    """
    Render the home page.
    """
    try:
        return render_template('index.html')
    except Exception as e:
        return str(e)

@app.route('/get', methods=['POST'])
def get_bot_response():
    """
    Get the bot's response to the user's input.
    """
    user_text = request.get_json().get('msg')
    formatted_prompt = format_prompt(user_text)
    print("User input:", user_text)
    print("Formatted prompt:", formatted_prompt)
    
    start_time = time.time()  # Start timing the response generation

    try:
        # Tokenize the input with truncation and padding
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024)
        prompt_length = inputs.input_ids.shape[1]

        # Log the length of the prompt
        print(f"Prompt length: {prompt_length}")

        # Calculate the maximum response length to stay within the 2048 token limit
        max_response_length = min(200, 2048 - prompt_length)
        if max_response_length <= 0:
            return jsonify({"error": "Input exceeds the maximum token limit."})

        # Log the max response length
        print(f"Max response length: {max_response_length}")

        # Generate the response
        response = model.generate(
            inputs['input_ids'], 
            attention_mask=inputs['attention_mask'],
            max_length=prompt_length + max_response_length,  # Ensure the total length doesn't exceed 2048 tokens
            min_length=prompt_length + 50,  # Ensure the response is meaningful
            pad_token_id=tokenizer.pad_token_id,  # Set pad token id
            no_repeat_ngram_size=2, 
            num_return_sequences=1, 
            temperature=0.50,  # Adjusted for more coherent output
            top_p=0.80  # Adjust top_p for more relevant results
        )

        # Log the generated response
        print("Response generated")

        generated_text = tokenizer.decode(response[0], skip_special_tokens=True)
        processed_response = post_process_response(generated_text)
        
        end_time = time.time()  # End timing the response generation
        response_time = end_time - start_time
        print(f"Time taken for response: {response_time} seconds")
        
        return jsonify({"response": processed_response, "response_time": response_time})
    except Exception as e:
        print("Error during model generation:", str(e))
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=False)
    