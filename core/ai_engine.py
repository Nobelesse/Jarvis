import requests

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_API_URL,
    OPENROUTER_MODEL,
)


SYSTEM_PROMPT = """
You are Jarvis, a helpful personal laptop assistant.

Rules:
- Keep answers clear, helpful, and suitable for speaking aloud.
- Use short paragraphs.
- Do not claim that you opened apps, changed files, or controlled the laptop.
- Be honest when you are uncertain.
- Address the user as Boss occasionally, but do not overuse it.
""".strip()


def get_ai_response(user_message):
    user_message = str(user_message).strip()

    if not user_message:
        return "I did not hear a question clearly."

    if not OPENROUTER_API_KEY:
        return (
            "OpenRouter is not configured. "
            "Please add your OpenRouter API key to the dot env file."
        )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "X-Title": "Jarvis",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        "temperature": 0.7,
        "max_tokens": 250,
    }

    try:
        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=45,
        )

        if response.status_code != 200:
            return (
                "I could not reach my online AI brain right now. "
                "Please check your internet connection or OpenRouter key."
            )

        data = response.json()

        choices = data.get("choices", [])

        if not choices:
            return "I received an empty response from my online AI brain."

        message = choices[0].get("message", {})
        content = message.get("content", "")

        if isinstance(content, list):
            text_parts = []

            for part in content:
                if isinstance(part, dict):
                    text = part.get("text", "")
                    if text:
                        text_parts.append(str(text))

            content = " ".join(text_parts)

        content = str(content).strip()

        if not content:
            return "I received an empty response from my online AI brain."

        return content

    except requests.Timeout:
        return "The online AI request took too long. Please try again."

    except requests.RequestException:
        return (
            "I cannot connect to OpenRouter right now. "
            "Please check your internet connection."
        )

    except ValueError:
        return "I received an invalid response from the online AI service."

    except Exception:
        return "Something unexpected happened while contacting my online AI brain."