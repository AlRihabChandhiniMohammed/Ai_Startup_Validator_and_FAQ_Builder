import os
import requests
import json
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS
load_dotenv()

app = Flask(__name__)
CORS(app) 
@app.route('/')
def home():
    """Simple endpoint to confirm the backend is running."""
    return "AI Startup Validator & FAQ Builder Backend is running!"

@app.route('/validate_startup', methods=['POST'])
def validate_startup():
    """
    Receives startup details, calls NVIDIA AI for validation,
    and returns the assessment.
    """
    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
    if not NVIDIA_API_KEY:
        return jsonify({"error": "NVIDIA_API_KEY is not set in the .env file."}), 500
    NVIDIA_MODEL_NAME = "meta/llama3-8b-instruct" 
    NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

    try:
        data = request.get_json()
        startup_name = data.get('startupName')
        description = data.get('description')
        target_market = data.get('targetMarket')
        business_model = data.get('businessModel')
        competitive_advantage = data.get('competitiveAdvantage')

        if not all([startup_name, description, target_market, business_model, competitive_advantage]):
            return jsonify({"error": "Missing one or more required startup details."}), 400

        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }

        prompt = f"""
        You are an expert startup validator and business analyst. Your task is to evaluate the following startup idea comprehensively.

        Startup Name: {startup_name}
        Description: {description}
        Target Market: {target_market}
        Business Model: {business_model}
        Competitive Advantage: {competitive_advantage}

        Please provide a detailed assessment structured as follows, using Markdown for clear formatting:

        **Startup Rating:** [X/10] - Provide a concise justification for this rating, highlighting key strengths and weaknesses.
        **Sentiment Analysis:** [Overall sentiment of the idea's viability, e.g., 'Highly Positive', 'Positive', 'Neutral', 'Slightly Negative', 'Negative'] - Explain the reasons behind this sentiment based on the provided details, considering market trends and potential.
        **Suggestions for Improvement:**
        * Suggestion 1: Explain how this improves the idea, focusing on market fit, scalability, technical feasibility, or execution strategy.
        * Suggestion 2: Another actionable suggestion with explanation.
        * Suggestion 3: A third practical suggestion (if applicable).
        **Relevant Ideas & References:**
        * Idea 1 Name: Brief description of a similar, complementary, or adjacent business idea. (If relevant, include a real or hypothetical URL reference, e.g., 'https://www.example.com/related_project')
        * Idea 2 Name: Brief description. (If relevant, include a real or hypothetical URL reference)
        * Idea 3 Name: Brief description. (If relevant, include a real or hypothetical URL reference)
        """

        payload = {
            "model": NVIDIA_MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 1024,
            "stream": False
        }

        nvidia_response = requests.post(NVIDIA_API_URL, headers=headers, json=payload, timeout=60)
        nvidia_response.raise_for_status()
        response_data = nvidia_response.json()

        if response_data and "choices" in response_data and response_data["choices"]:
            ai_text = response_data["choices"][0]["message"]["content"]
            return jsonify({"success": True, "ai_response": ai_text})
        else:
            print(f"ERROR: No valid response from NVIDIA AI for validator. Raw response: {json.dumps(response_data, indent=2)}")
            return jsonify({"error": "No valid response from NVIDIA API.", "raw_response": response_data}), 500

    except requests.exceptions.RequestException as e:
        print(f"ERROR: NVIDIA API request failed for validator: {e}")
        return jsonify({"error": f"NVIDIA API request failed: {e}", "details": str(e)}), 500
    except json.JSONDecodeError:
        response_text = nvidia_response.text if 'nvidia_response' in locals() else 'No response from API'
        print(f"ERROR: Invalid JSON response from NVIDIA API for validator. Raw response: {response_text}")
        return jsonify({"error": "Invalid JSON response from NVIDIA API.", "raw_response": response_text}), 500
    except Exception as e:
        print(f"ERROR: An unexpected error occurred for validator: {e}")
        return jsonify({"error": f"An unexpected error occurred: {e}", "details": str(e)}), 500


@app.route('/generate_faq', methods=['POST'])
def generate_faq():
    """
    Receives startup details, uses NVIDIA AI to generate FAQs,
    and returns the generated FAQ section.
    """
    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

    if not NVIDIA_API_KEY:
        return jsonify({"error": "NVIDIA_API_KEY is not set in the .env file."}), 500
    NVIDIA_MODEL_NAME = "meta/llama3-8b-instruct" 
    NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

    try:
        data = request.get_json()
        startup_name = data.get('startupName')
        startup_description = data.get('startupDescription')

        if not all([startup_name, startup_description]):
            return jsonify({"error": "Missing startup name or description."}), 400

        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }

        prompt = f"""
        You are an expert AI assistant specializing in creating comprehensive Frequently Asked Questions (FAQ) sections for new startups.
        Your goal is to anticipate common questions potential customers, investors, or users might have about the startup.

        Startup Name: {startup_name}
        Startup Description: {startup_description}

        Generate a list of 5-8 common and insightful FAQs with concise answers.
        Format your response clearly using Markdown, with each question as a bold heading and the answer following directly.

        Example Format:
        **Q: What is [Startup Name]?**
        A: [Concise answer about what it does and its core value proposition.]

        **Q: How does [Startup Name] work?**
        A: [Explanation of the basic mechanics or user flow.]

        --- Start Generating FAQs ---
        """

        payload = {
            "model": NVIDIA_MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 1000,
            "stream": False
        }

        nvidia_response = requests.post(NVIDIA_API_URL, headers=headers, json=payload, timeout=60)
        nvidia_response.raise_for_status()
        response_data = nvidia_response.json()

        if response_data and "choices" in response_data and response_data["choices"]:
            ai_text = response_data["choices"][0]["message"]["content"]
            return jsonify({"success": True, "faq_content": ai_text})
        else:
            print(f"ERROR: No valid response from NVIDIA AI for FAQ. Raw response: {json.dumps(response_data, indent=2)}")
            return jsonify({"error": "No valid response from NVIDIA API.", "raw_response": response_data}), 500

    except requests.exceptions.RequestException as e:
        print(f"ERROR: NVIDIA API request failed for FAQ: {e}")
        return jsonify({"error": f"NVIDIA API request failed: {e}", "details": str(e)}), 500
    except json.JSONDecodeError:
        response_text = nvidia_response.text if 'nvidia_response' in locals() else 'No response from API'
        print(f"ERROR: Invalid JSON response from NVIDIA API for FAQ. Raw response: {response_text}")
        return jsonify({"error": "Invalid JSON response from NVIDIA API.", "raw_response": response_text}), 500
    except Exception as e:
        print(f"ERROR: An unexpected error occurred for FAQ: {e}")
        return jsonify({"error": f"An unexpected error occurred: {e}", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
