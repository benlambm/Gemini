from google import genai

client = genai.Client()

print("Fetching available models...")
try:
    # The SDK method might vary slightly depending on version, 
    # but client.models.list() is standard for the newer SDKs.
    # We iterate and print to see what's actually available to your key.
    for model in client.models.list():
        print(f"- {model.name}")
except Exception as e:
    print(f"Error listing models: {e}")
