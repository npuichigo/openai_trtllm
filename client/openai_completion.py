import pprint

from openai import OpenAI

client = OpenAI(base_url="http://localhost:3000/v1", api_key="test")

result = client.completions.create(
    model="ensemble",
    prompt="Say this is a test",
)
pprint.pprint(result)
