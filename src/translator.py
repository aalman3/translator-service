import openai
import time

# Use your actual API Key and Endpoint
openai.api_key = os.getenv("OPENAI_API_KEY") 
openai.api_base = "https://barbol.openai.azure.com/"  # Your endpoint base URL
openai.api_type = "azure"
openai.api_version = "2024-08-01-preview"
deployment_name = "barbol-gpt-4"

def get_translation(post: str) -> str:
    # Define the translation prompt in a conversational format
    messages = [
        {"role": "system", "content": "You are a helpful assistant that translates text to English."},
        {"role": "user", "content": f"Translate the following text to English: {post}"}
    ]

    wait_time = 5

    while True:
        try:
            # Use the ChatCompletion endpoint with the gpt-4 model
            response = openai.ChatCompletion.create(
                engine="barbol-gpt-4",
                messages=messages,
                max_tokens=100,
                temperature=0.2,
            )

            # Extract and return the translation from the response
            translation = response.choices[0].message['content'].strip()
            return translation

        except openai.error.RateLimitError:
            print(f"Rate limit exceeded. Waiting for {wait_time} seconds before retrying...")
            time.sleep(wait_time)
        except Exception as e:
            print("Error in translation:", e)
            return ""

def get_language(post: str) -> str:
    # Define the prompt to determine the exact language of the text
    messages = [
        {"role": "system", "content": "You are a helpful assistant that detects the language of the given text and returns the name of the language, starting with a capital letter and a one word response(eg. English, Spanish, Arabic, Russian)."},
        {"role": "user", "content": f"Identify the language of the following text: {post}"}
    ]

    wait_time = 5

    while True:
        try:
            # Use the ChatCompletion endpoint to determine language
            response = openai.ChatCompletion.create(
                engine="barbol-gpt-4",
                messages=messages,
                max_tokens=10,
                temperature=0.0,
            )

            # Extract and return the detected language from the response
            detected_language = response.choices[0].message['content'].strip()
            return detected_language

        except openai.error.RateLimitError:
            print(f"Rate limit exceeded. Waiting for {wait_time} seconds before retrying...")
            time.sleep(wait_time)
        except Exception as e:
            print("Error in language detection:", e)
            return ""

def translate_content(post: str) -> str:
    """
    Detects the language of the text and translates it into English if it's not already in English.
    """
    try:
        # Step 1: Detect the language of the text
        detected_language = get_language(post)
        print(f"Detected Language: {detected_language}")

        # Step 2: Translate if the text is not in English
        if detected_language != "English":
            print(f"Translating text from {detected_language} to English...")
            return get_translation(post)
        else:
            print("Text is already in English. No translation needed.")
            return post

    except Exception as e:
        print("Error in translate_content:", e)
        return ""




