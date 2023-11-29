import pprint

import openai

openai.api_key = "test"
openai.api_base = "http://localhost:3000/v1"

result = openai.Completion.create(
    model="ensemble",
    prompt="Say this is a test",
)
pprint.pprint(result)
