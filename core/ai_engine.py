import requests

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_API_URL,
    OPENROUTER_MODEL,
    OPENROUTER_TIMEOUT_SECONDS,
    openrouter_is_configured,
)


SYSTEM_PROMPT = """
You are Jarvis, a helpful personal assistant running locally on a Windows laptop.

Rules:
- Give clear and helpful answers.
- Keep answers short enough for spoken conversation unless more detail is requested.
- Do not claim you performed an action on the laptop unless the local program
  confirms that action.
- Address the user as Boss occasionally, but do not overuse it.
""".strip()


def _get_error_message(response):
    try:
        data = response.json()
    except ValueError:
        return ""

    error = data.get("error", {})

    if isinstance(error, dict):
        return str(error.get("message", "")).strip()

    return str(error).strip()


def _extract_answer(response_data):
    choices = response_data.get("choices", [])

    if not choices:
        return ""

    first_choice = choices[0]

    if not isinstance(first_choice, dict):
        return ""

    message = first_choice.get("message", {})

    if not isinstance(message, dict):
        return ""

    content = message.get("content", "")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, dict):
                text = item.get("text", "")

                if text:
                    text_parts.append(str(text))

        return " ".join(text_parts).strip()

    return str(content).strip()


def get_ai_response(user_message):
    question = str(user_message).strip()

    if not question:
        return "I did not hear a clear question, Boss."

    if not openrouter_is_configured():
        return (
            "OpenRouter is not configured. "
            "Please add your OpenRouter API key to the dot env file."
        )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "X-Title": "Jarvis Local Assistant",
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
                "content": question,
            },
        ],
        "max_tokens": 350,
    }

    try:
        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=OPENROUTER_TIMEOUT_SECONDS,
        )

    except requests.exceptions.ConnectionError:
        return (
            "I could not connect to OpenRouter. "
            "Please check your internet connection."
        )

    except requests.exceptions.Timeout:
        return (
            "OpenRouter took too long to respond. "
            "Please try again."
        )

    except requests.exceptions.RequestException:
        return (
            "I had a network problem while contacting OpenRouter."
        )

    if response.status_code == 401:
        return (
            "Your OpenRouter API key was rejected. "
            "Please check the key in your dot env file."
        )

    if response.status_code == 429:
        return (
            "The OpenRouter free request limit has been reached. "
            "Please try again later."
        )

    if not response.ok:
        error_message = _get_error_message(response)

        if error_message:
            return f"OpenRouter could not process the request: {error_message}"

        return "OpenRouter could not process that request."

    try:
        response_data = response.json()
    except ValueError:
        return "I received an invalid response from OpenRouter."

    answer = _extract_answer(response_data)

    if not answer:
        return "I did not receive a usable answer from OpenRouter."

    return answer