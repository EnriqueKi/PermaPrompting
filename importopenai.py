import openai
import requests

openai.api_key = "YOUR_OPENAI_API_KEY"

def generate_image(prompt, filename="output.png"):
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="512x512"
    )
    image_url = response['data'][0]['url']
    img_data = requests.get(image_url).content
    with open(filename, 'wb') as handler:
        handler.write(img_data)
    print(f"Image saved as {filename}")

if __name__ == "__main__":
    user_prompt = input("Enter a prompt for the image: ")
    generate_image(user_prompt)