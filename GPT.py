from openai import OpenAI

class gpt:
    def __init__(self):
        self.client = OpenAI(api_key="aqui va el token")

    def get_completions(self,  prompt:str,model='gpt-3.5-turbo-16k-0613',temperature =0):
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model = model,
            messages = messages,
            temperature = temperature
        )
        return response.choices[0].message.content