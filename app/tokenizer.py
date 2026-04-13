import string
from collections import Counter
import json


class Tokenizer:
    def __init__(self,vocab_dict=None):
        self.max_vocab_size = 10000
        
        self.PAD_TOKEN = "<PAD>"
        self.UNK_TOKEN = "<UNK>"
        
        
        if vocab_dict:
            self.word_to_id = vocab_dict
            self.id_to_word = { int(idx): word for word,idx in vocab_dict.items()}
        else:
            
            self.word_to_id = {
                self.PAD_TOKEN: 0,
                self.UNK_TOKEN: 1
            }
            
            # 3. Do the same for the reverse map
            self.id_to_word = {
                0: self.PAD_TOKEN,
                1: self.UNK_TOKEN
            }
    
    def clean_text(self,text:str) -> str:
        
        text = text.lower()
        
        
        #Remove punctuation using a transalation table
        # This creates a mapping where every punctuation char is replaced by None (deleted)
        
        table = str.maketrans('','',string.punctuation)
        text = text.translate(table)
        
        return text
    
    
    def tokenize(self,text:str)->list:
        clean = self.clean_text(text)
        
        
        tokens = clean.split()
        
        return tokens
    
    
    def build_vocab(self,sentences:list):
        
        word_counts = Counter()
        
        for sentence in sentences:
            tokens = self.tokenize(sentence)
            
            word_counts.update(tokens)
            
        common_words = word_counts.most_common(self.max_vocab_size -2)
        
        
        for word ,count in common_words:
            if word not in self.word_to_id:
                new_id = len(self.word_to_id)
                self.word_to_id[word] = new_id
                self.id_to_word[new_id] =word
                
                
                
    def encode(self,sentences:list)->list:
        
        # tokens = []
        # encoded_list = []
        
        # for sentence in sentences:
        #     tokens.append(self.tokenize(sentence))
            
        # for token in tokens:
            
        #     if not self.word_to_id.get(token):
        #         encoded_list.append(self.UNK_TOKEN)
        #     else:
        #         encoded_list.append(self.word_to_id.get(token))
                
        # return encoded_list
        
        encoded_list = []
        
        print(f"this is are sentences we are expecting :{sentences}")
        
        for sentence in sentences:
            
            words = self.tokenize(sentence)
            
            sentence_ids = []
            
            
            for word in words:
                # 2. Look up the ID, defaulting to the UNK ID (1) if not found
                word_id = self.word_to_id.get(word,self.word_to_id[self.UNK_TOKEN])
                
                sentence_ids.append(word_id)
                
            encoded_list.append(sentence_ids)
            
        return encoded_list
    
    
    def decode(self,token_ids:list)->list:
        
        decoded_sentences = []
        
        for sentence_ids in token_ids:
            words = []
            
            for token_id in sentence_ids:
                
                word = self.id_to_word.get(token_id,self.UNK_TOKEN)
                
                words.append(word)
                
            decoded_sentences.append(words)
            
        return decoded_sentences
    
    
    def save(self,filepath="./../data/vocab.json"):
        with open(filepath,"w") as f:
            json.dump(self.word_to_id,f)
            
            
        
    def load(self,filepath="./../data/vocab.json"):
        with open(filepath,"r") as f:
            self.word_to_id = json.load(f)
            
            self.id_to_word = {idx: word for word, idx in self.word_to_id.items()}
            
        
    
    
    