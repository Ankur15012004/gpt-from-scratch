"""
Tokenizer wrapper for GPT-2 using tiktoken
Provides a clean interface for text encoding/decoding
"""

import tiktoken

class GPTTokenizer:
    """
    A wrapper around tiktoken's GPT-2 tokenizer 

    Why we need this wrapper:
    - Consistent interface across our project 
    - Easy to add speacial tokens if needed 
    - Handles edge cases and provides utilities 
    
    """

    def __init__(self,model_name: str="gpt2"):
        """
        Initialize the tokenizer 

        Args:
            model_name: Which tiktoken encoding to use (gpt2)

        Why GPT-2: It's well-tested, handles most text wellm and is what most small 
        language models are trained with 
        
        """
        self.model_name=model_name
        self.tokenizer=tiktoken.get_encoding(model_name)

        # Store vocab size- we'll need this for our embedding layer 
        self.vocab_size=self.tokenizer.n_vocab

        print(f"Loaded {model_name} tokenizer with vocab size: {self.vocab_size}")

    def encode(self, text: str) -> list[int]:
        """
            Convert text to token IDs

            Args:
                text: Input text string 

            Returns:
                List of token IDs (integers)

            Examples:
                "Hello world" ->[15496,995]
        
        """
        if not isinstance(text,str):
            raise ValueError(f"Expected string, got {type(text)}")
        
        if not text.strip():
            return []
        
        token_ids=self.tokenizer.encode(text)

        return token_ids
    
    def decode(self, token_ids: list[int])->str:
        """
        Convert token IDs back to text
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text string
            
        Example:
            [15496, 995] -> "Hello world"
        """
        if not isinstance(token_ids, (list,tuple)):
            raise ValueError(f"Expected list or tuple, got {type(token_ids)}")
        
        if not token_ids:
            return ""
        
        text=self.tokenizer.decode(token_ids)

        return text
    
    def encode_batch(self, texts: list[str])->list[list[int]]:
        """Encode multiple texts at once"""

        return [self.encode(text) for text in texts]
    
    def decode_batch(self, token_ids_batch: list[list[int]]) -> list[str]:
        """Decode multiple token sequences at once"""
        return [self.decode(token_ids) for token_ids in token_ids_batch]
    
    def get_vocab_size(self) -> int:
        """Return the vocabulary size"""
        return self.vocab_size
    
    def token_to_string(self, token_id: int) -> str:
        """
        Convert a single token ID to its string representation
        
        Useful for debugging and understanding what each token represents
        """
        return self.tokenizer.decode([token_id])
    
    def get_token_info(self, text: str) -> dict:
        """
        Get detailed information about how text is tokenized
        
        Useful for debugging and understanding tokenization
        """
        token_ids = self.encode(text)
        tokens = [self.token_to_string(tid) for tid in token_ids]
        
        return {
            'text': text,
            'token_ids': token_ids,
            'tokens': tokens,
            'num_tokens': len(token_ids)
        }
    
def test_tokenizer():
    """
    Test our tokenizer implementation
    
    This helps us verify everything works correctly
    """
    print("🧪 Testing Tokenizer...")
    
    # Initialize tokenizer
    tokenizer = GPTTokenizer()
    
    # Test basic encoding/decoding
    test_text = "Hello, world! This is a test."
    print(f"\nOriginal text: '{test_text}'")
    
    # Encode
    token_ids = tokenizer.encode(test_text)
    print(f"Token IDs: {token_ids}")
    
    # Decode
    decoded_text = tokenizer.decode(token_ids)
    print(f"Decoded text: '{decoded_text}'")
    
    # Check if they match
    assert test_text == decoded_text, "Encoding/decoding mismatch!"
    print("✅ Basic encode/decode test passed!")
    
    # Test detailed info
    info = tokenizer.get_token_info(test_text)
    print(f"\nDetailed tokenization:")
    for i, (token_id, token) in enumerate(zip(info['token_ids'], info['tokens'])):
        print(f"  {i}: {token_id} -> '{token}'")
    
    print(f"\nVocab size: {tokenizer.get_vocab_size()}")
    print("🎉 All tokenizer tests passed!")

# Run tests if this file is executed directly
if __name__ == "__main__":
    test_tokenizer()

      

    


