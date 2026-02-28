from google import genai

client = genai.Client()

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Create a short poem about the beauty of nature.",
)

print(response.text)
