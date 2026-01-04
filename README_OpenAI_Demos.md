# OpenAI Demo Notebooks

## Overview
This folder contains notebooks demonstrating usage of the OpenAI API for chat completions, function calling, model listing, and simple examples of interacting with the API using `openai` Python client.

## Demos Included
- `openaidemo1.ipynb`: Basic ChatCompletion examples, model listing, and simple prompts
- `openaidemo2.ipynb`: Function calling examples and integrating external APIs

## Requirements
- `openai` Python package
- `.env` file with `OPENAI_API_KEY` set
- `requests` for external API calls (in function calling demo)

## Example Usage
```python
from dotenv import load_dotenv
import openai
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

prompt = "Hello, how are you?"
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role":"system","content":"You are a helpful assistant."},
              {"role":"user","content":prompt}]
)
print(response["choices"][0]["message"]["content"])
```

## Author
**Ranit Pal** - Data Science & AI Learning Project
- GitHub: [@ranit57](https://github.com/ranit57)
- Email: ranitpal57@gmail.com

---
**Last Updated**: January 2026
