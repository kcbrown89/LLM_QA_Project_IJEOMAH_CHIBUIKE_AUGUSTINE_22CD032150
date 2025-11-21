import os
import string
from google import genai
from dotenv import load_dotenv
from flask import Flask, request, render_template

# Load environment variables
load_dotenv()

app = Flask(__name__)

# --- Preprocessing Function ---
def preprocess_text(text: str) -> str:
    """Applies basic preprocessing: lowercasing, and punctuation removal."""
    if not text:
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    cleaned_tokens = text.split()
    return " ".join(cleaned_tokens)

@app.route('/', methods=['GET', 'POST'])
def index():
    question = ""
    processed_question = ""
    llm_response = ""
    error_message = ""

    if request.method == 'POST':
        question = request.form.get('question')
        
        if not question:
            error_message = "Please enter a question."
            return render_template('index.html', error=error_message)

        # 1. Process the question
        processed_question = preprocess_text(question)

        # 2. Setup LLM API call
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            # IMPORTANT: For deployment, ensure the API key is passed correctly 
            # as an environment variable in your hosting platform (Render/Streamlit Cloud).
            error_message = "Server Error: LLM API key not configured on the server."
            return render_template('index.html', question=question, processed_question=processed_question, error=error_message)

        try:
            client = genai.Client(api_key=api_key)
            
            # Construct the prompt
            system_instruction = "You are a helpful Question Answering bot. Provide a concise and accurate answer to the user's question."
            prompt = f"Question: {processed_question}"

            # 3. Send to LLM API
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config={'system_instruction': system_instruction}
            )
            
            # 4. Display the generated answer
            llm_response = response.text.strip()
            
        except Exception as e:
            llm_response = f"An error occurred while communicating with the LLM API. ({e})"
            print(f"LLM API Error: {e}")
            
    # Render the template with the results
    return render_template('index.html', 
                           question=question, 
                           processed_question=processed_question, 
                           llm_response=llm_response,
                           error=error_message)

if __name__ == '__main__':
    # Use host='0.0.0.0' for local testing and for deployment compatibility
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)