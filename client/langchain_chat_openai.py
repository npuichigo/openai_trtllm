from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage


chat = ChatOpenAI(openai_api_base="http://localhost:3000/v1",
                  openai_api_key="test", model_name="ensemble",
                  max_tokens=100)

messages = [
    SystemMessage(content="You're a helpful assistant"),
    HumanMessage(content="What is the purpose of model regularization?"),
]

result = chat.invoke(messages)
print(result.content)
