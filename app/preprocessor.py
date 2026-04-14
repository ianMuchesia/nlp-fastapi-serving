import torch



class TextPreprocessor:
    def  __init__(self,tokenizer):
        self.tokenizer = tokenizer
        
        
    def clean(self, text:str) -> str:
        return text.strip().lower()
    
    def __call__(self, texts:list[str])->torch.Tensor:
        
        cleaned_texts = [ self.clean(text) for text in texts]
        tokens = self.tokenizer.encode(cleaned_texts)
            
        max_len = max(len(t) for t in tokens)
        padded_tokens = [ t + [0] * (max_len-len(t)) for t in tokens]
        input_tensor = torch.tensor(padded_tokens).long()
        
        return input_tensor