
class MultimodalModel:
    def __init__(self, model_name="llava:7b"):
        self.model_name = model_name
        import ollama
        self.client = ollama
        self.history = []

    def chat(self, prompt, images=None):
        body = {
            "model": self.model_name,
            "prompt": prompt
        }
        if images:
            body["images"] = images
        response = self.client.generate(body)
        return response['response']
