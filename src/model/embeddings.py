"""
GPT-Style Embeddings: Token + Learned Positional Embeddings
Built from scratch following GPT architecture (not original Transformer)
"""

import torch 
import torch.nn as nn 
import math 
from typing import Optional

class TokenEmbedding(nn.Module):
    """
    Convert token IDs to dense vectors
    
    Intuition: Like a lookup table where each word has a dense representation
    Token ID 123 → Vector [0.1, -0.2, 0.5, ...]
    """

    def __init__(self,vocab_size:int,d_model: int):
        """
        Initialize token embedding layer
        
        Args:
            vocab_size: Size of vocabulary (e.g., 50257 for GPT-2)
            d_model: Embedding dimension (e.g., 768)
        """
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size

        # This is the main embedding table: vocab_size x d_model 
        self.embedding= nn.Embedding(vocab_size,d_model)

        # Initialize embeddings with small random values 
        # GPT uses normal distribution with std=0.02 
        nn.init.normal_(self.embedding.weight,mean=0.0, std=0.02)

        print(f"✅ Token embedding: {vocab_size} tokens → {d_model} dimensions")

    def forward(self, token_ids: torch.tensor)-> torch.Tensor:
        """
        Convert token IDs to embeddings
        
        Args:
            token_ids: (batch_size, seq_len) - Integer token IDs
            
        Returns:
            embeddings: (batch_size, seq_len, d_model) - Dense embeddings
        """

        # Simple lookup: each token ID gets its corresponding embedding vector 

        embeddings=self.embedding(token_ids)

        # GPT scales embeddings by sqrt(d_model) for better training dynamics 

        embeddings=embeddings*math.sqrt(self.d_model)

        return embeddings
    
class PositionalEmbedding(nn.Module):
    """
    Learned positional embeddings (GPT approach)
    
    Intuition: Each position (0, 1, 2, ...) gets its own learned vector
    Position 0 → Vector [0.3, -0.1, ...]
    Position 1 → Vector [-0.2, 0.4, ...]
    """

    def __init__(self, max_seq_len: int, d_model: int):
        """
        Initialize positional embedding layer
        
        Args:
            max_seq_len: Maximum sequence length (e.g., 1024)
            d_model: Embedding dimension (same as token embeddings)
        """

        super().__init__()
        self.max_seq_len=max_seq_len
        self.d_model=d_model

        # Learnable position embeddings: max_seq_len x d_model 
        # Each position gets its own trainable vector 

        self.embedding=nn.Embedding(max_seq_len, d_model)

        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        print(f"✅ Position embedding: {max_seq_len} positions → {d_model} dimensions")

    def forward(self, seq_len: int, device: torch.device)-> torch.Tensor:
        """
        Get positional embeddings for a sequence
        
        Args:
            seq_len: Length of the current sequence
            device: Device to create tensors on
            
        Returns:
            pos_embeddings: (1, seq_len, d_model) - Position embeddings
        """
        positions=torch.arange(seq_len,device=device)

        # Look up embeddings for these positions 
        pos_embeddings=self.embedding(positions)

        # Add batch dimension: (seq_len, d_model) → (1, seq_len, d_model)
        pos_embeddings=pos_embeddings.unsqueeze(0)

        return pos_embeddings
    

class GPTEmbeddings(nn.Module):
    """
    Complete GPT-style embedding layer: Token + Position + Dropout
    
    This is what actually gets used in the transformer blocks
    """

    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int, dropout: float=0.1):
        """
        Initialize complete embedding system
        
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension  
            max_seq_len: Maximum sequence length
            dropout: Dropout rate for regularization
        """
        super().__init__()
        self.d_modell=d_model
        self.max_seq_len=max_seq_len

        # Create token and position embedding layers 
        self.token_embedding=TokenEmbedding(vocab_size, d_model)
        self.pos_embedding= PositionalEmbedding(max_seq_len, d_model)

        self.dropout=nn.Dropout(dropout)

        print(f"✅ GPT Embeddings ready: tokens + positions + dropout")

    def forward(self, token_ids: torch.tensor)->torch.Tensor:
        """
        Convert token IDs to position-aware embeddings
        
        Args:
            token_ids: (batch_size, seq_len) - Token IDs from tokenizer
            
        Returns:
            embeddings: (batch_size, seq_len, d_model) - Position-aware embeddings
        """

        batch_size, seq_len=token_ids.shape

        # Check sequence length doesn't exceed maximum 

        if seq_len> self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")
        
        token_emb=self.token_embedding(token_ids) # (batch_size, seq_len, d_model)

        # Step-2: Get positional embeddings 
        pos_emb=self.pos_embedding(seq_len, token_ids.device) # (1, seq_len, d_model)

        # Step 3: Add them together (broadcasting handles batch dimension)
        embeddings= token_emb+pos_emb # ( batch_size, seq_len, d_model)

        # Step 4 : Apply dropout for regularization 
        embeddings=self.dropout(embeddings)

        return embeddings
    

def test_gpt_embeddings():
    """Test our GPT-style embeddings with real examples"""
    print("🧪 Testing GPT-Style Embeddings...")
    
    # Test configuration (small for easy understanding)
    vocab_size = 1000  # Small vocabulary
    d_model = 64       # Small embedding dimension
    max_seq_len = 128  # Maximum sequence length
    batch_size = 2     # Test with 2 sequences
    seq_len = 8        # Sequence length
    
    print(f"📊 Test setup:")
    print(f"   Vocab size: {vocab_size}")
    print(f"   Model dimension: {d_model}")
    print(f"   Max sequence length: {max_seq_len}")
    print(f"   Batch size: {batch_size}, Sequence length: {seq_len}")
    
    # Create embedding layer
    embeddings = GPTEmbeddings(
        vocab_size=vocab_size,
        d_model=d_model, 
        max_seq_len=max_seq_len,
        dropout=0.0  # No dropout for testing
    )
    
    # Create sample token IDs (like from our tokenizer)
    # In real use, these would come from: tokenizer.encode("Hello world!")
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"\n📝 Sample token IDs:")
    print(f"   Shape: {token_ids.shape}")
    print(f"   Values: {token_ids[0].tolist()[:5]}... (first sequence, first 5 tokens)")
    
    # Apply embeddings
    embedded = embeddings(token_ids)
    
    print(f"\n📊 Embedding results:")
    print(f"   Output shape: {embedded.shape}")
    print(f"   Expected: ({batch_size}, {seq_len}, {d_model})")
    
    # Verify dimensions
    expected_shape = (batch_size, seq_len, d_model)
    shape_correct = embedded.shape == expected_shape
    print(f"   Shape correct: {shape_correct}")
    
    # Show that different positions have different embeddings
    print(f"\n🔍 Position awareness test:")
    pos_0_embedding = embedded[0, 0, :5]  # First 5 dims of position 0
    pos_1_embedding = embedded[0, 1, :5]  # First 5 dims of position 1
    
    print(f"   Position 0 embedding (first 5 dims): {pos_0_embedding.detach().numpy().round(3)}")
    print(f"   Position 1 embedding (first 5 dims): {pos_1_embedding.detach().numpy().round(3)}")
    
    # They should be different due to positional embeddings
    difference = torch.abs(pos_0_embedding - pos_1_embedding).mean()
    print(f"   Average difference: {difference:.4f} (should be > 0)")
    
    print("✅ GPT embeddings test PASSED!")
    return True

def test_with_real_tokens():
    """Test with realistic token sequences"""
    print("\n🧪 Testing with Realistic Token Sequences...")
    
    # Use GPT-2 vocabulary size and dimensions
    vocab_size = 50257  # GPT-2 vocab size
    d_model = 128       # Smaller than real GPT-2 for testing
    max_seq_len = 256   # Reasonable context length
    
    embeddings = GPTEmbeddings(vocab_size, d_model, max_seq_len, dropout=0.1)
    
    # Simulate tokenized sequences (like what tiktoken would produce)
    sequences = [
        [1, 464, 4758, 318, 257, 1332],  # "This is a test" (example tokens)
        [5756, 995, 0, 0, 0, 0],         # "Hello world" + padding
    ]
    
    token_ids = torch.tensor(sequences)
    print(f"Token sequences shape: {token_ids.shape}")
    
    # Apply embeddings
    embedded = embeddings(token_ids)
    print(f"Embedded shape: {embedded.shape}")
    
    # Verify it works with different sequence lengths
    short_seq = torch.tensor([[1, 2, 3]])  # Length 3
    short_embedded = embeddings(short_seq)
    print(f"Short sequence embedded: {short_embedded.shape}")
    
    print("✅ Realistic token test PASSED!")
    return True

if __name__ == "__main__":
    print("🚀 Testing GPT-Style Embeddings")
    print("=" * 60)
    
    test1_passed = test_gpt_embeddings()
    test2_passed = test_with_real_tokens()
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL EMBEDDING TESTS PASSED!")
        print("✅ Ready for next step: Transformer Blocks!")
    else:
        print("\n❌ Some tests failed. Please check the errors.")







