import base64
import os

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8081")


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"


def send_chat_completion_request(instruction, image_base64_url, base_url=BASE_URL):
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "max_tokens": 800,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_base64_url,
                            },
                        },
                    ],
                }
            ],
        },
    )

    if not response.ok:
        return f"Server error: {response.status_code} - {response.text}"

    return response.json()["choices"][0]["message"]["content"]


def main():
    # Configuration
    image_path = "/home/aashish/Pictures/english_handwriting.jpg"
    instruction = "What is in this image?"
    base_url = BASE_URL

    try:
        # Convert image to base64
        image_base64 = image_to_base64(image_path)

        # Send request and get response
        response = send_chat_completion_request(instruction, image_base64, base_url)

        # Print the response
        print("Response:", response)

    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
