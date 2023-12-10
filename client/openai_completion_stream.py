from sys import stdout

from openai import OpenAI

client = OpenAI(base_url="http://localhost:3000/v1", api_key="test")

response = client.completions.create(
    model="ensemble",
    prompt="This is a story of a hero who went",
    stream=True,
    max_tokens=50,
)
for event in response:
    if not isinstance(event, dict):
        event = event.model_dump()
    event_text = event["choices"][0]["text"]
    stdout.write(event_text)
    stdout.flush()
