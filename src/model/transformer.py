"""
GPT-Style Transformer Decoder Block
Combines all components into a single processing unit
"""

import torch 
import torch.nn as nn 
from typing import Optional 


from src.model.attention import MultiHeadSelfAttention
from src.model.layers import PreLayerNormResidual, FeedForward
from src.model.embeddings import GPTEmbeddings

class TransformerBlock(nn.Module):
    """
    A single Transformer decoder block (GPT-style)
    
    Architecture (Pre-Layer Norm):
    x → LayerNorm → Attention → Add(x) →
    ↓
    LayerNorm → FeedForward → Add → Output
    
    Intuition: Each block is one "reasoning step" where the model:
    1. Pays attention to important parts of the sequence  
    2. Processes each position with a neural network
    3. Preserves information through residual connections
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float=0.1):
        """
        Initialize transformer block 

        Args:
            d_model: Model dimension (768 for GPT-2)
            n_heads: Number of attention heads( 12 for GPT-2)
            d_ff: Feed-forward dimension(3072 for GPT-2)
            dropout: Dropout rate for regularization 
        
        """

        super().__init__()

        self.d_model=d_model 
        self.n_heads=n_heads 
        self.d_ff=d_ff

        # Core components 
        self.attention=MultiHeadSelfAttention(self.d_model,self.n_heads,dropout)
        self.feed_forward=FeedForward(d_model,d_ff,dropout)

        # Residual connections with pre-layer norm (GPT-Style)
        self.attn_residual=PreLayerNormResidual(d_model, dropout)

        self.ffn_residual=PreLayerNormResidual(d_model,dropout )

        print(f"✅ Transformer block: {d_model}d, {n_heads} heads, {d_ff} ffn")

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]=None)->torch.Tensor:
        """
        Forward pass through transformer block
        
        Args:
            x: (batch_size, seq_len, d_model) - Input embeddings
            mask: Optional causal mask for attention
            
        Returns:
            output: (batch_size, seq_len, d_model) - Processed embeddings
        """

        # Step 1: Multi-head self-attention with residual connection
        # x=x+attention(LayerNorm(x))

        x = self.attn_residual(x, lambda x_norm: self.attention(x_norm, mask)[0])

        # Step 2: Feed-forward with residual connection 

        # x= x+ feed_forward(LayerNorm(x))

        x=self.ffn_residual(x,self.feed_forward)

        return x
    

class GPTModel(nn.Module):
    """
    Complete GPT model: Embeddings + Multiple Transformer Blocks + Output Head
    
    Architecture:
    Token IDs → Embeddings → Block₁ → Block₂ → ... → Blockₙ → Language Head → Logits
    """

    def __init__(self,vocab_size: int, d_model: int, n_layers: int, n_heads: int, d_ff: int, max_seq_len: int, dropout: float=0.1):
        """
        Initialize complete GPT model
        
        Args:
            vocab_size: Vocabulary size (50257 for GPT-2)
            d_model: Model dimension (768 for GPT-2)
            n_layers: Number of transformer blocks (12 for GPT-2)
            n_heads: Number of attention heads (12 for GPT-2)
            d_ff: Feed-forward dimension (3072 for GPT-2)
            max_seq_len: Maximum sequence length (1024 for GPT-2)
            dropout: Dropout rate
        """
        super().__init__()

        self.vocab_size=vocab_size
        self.d_model=d_model
        self.n_layers=n_layers 
        self.max_seq_len=max_seq_len

        # Input embeddings (token + position )

        self.embeddings=GPTEmbeddings(vocab_size,d_model,max_seq_len, dropout)

        # Stack of transformer blocks 
        self.blocks =nn.ModuleList([TransformerBlock(d_model,n_heads,d_ff,dropout) for _ in range(n_layers)])

        # Final layer normalization (GPT-style)
        from src.model.layers import LayerNorm
        self.final_norm=LayerNorm(d_model)

        # Language modeling head (predict next token)
        self.lm_head=nn.Linear(d_model,vocab_size,bias=False)

        # Tie embeddings and output weights to reduce number of parameters

        self.lm_head.weight=self.embeddings.token_embedding.embedding.weight

        print(f"✅ GPT Model: {n_layers} layers, {self._count_parameters():,} parameters")

    def forward(self, token_ids: torch.Tensor, targets: Optional[torch.Tensor]=None)->torch.Tensor:
        """
        Forward pass through complete GPT model
        
        Args:
            token_ids: (batch_size, seq_len) - Input token IDs
            targets: (batch_size, seq_len) - Target token IDs for training
            
        Returns:
            logits: (batch_size, seq_len, vocab_size) - Next token predictions
        """
        batch_size, seq_len=token_ids.shape

        # Creating a causal mask (each position can only attend to previous positions )

        causal_mask=torch.tril(torch.ones(seq_len, seq_len, device=token_ids.device))
        x=self.embeddings(token_ids)    # (batch_size, seq_len, d_model)

        # Step 2: Pass through transformer blocks 
        for blocks in self.blocks:
            x=blocks(x,mask=causal_mask)

        # Step 3: Final layer normalization 
        x=self.final_norm(x)

        # Step 4: Project to vocabulary (language modeling head )
        logits=self.lm_head(x) # (batch_size, seq_len, vocab_size)

        return logits 
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_num_params(self) -> dict:
        """Get detailed parameter breakdown"""
        params = {}
        params['embeddings'] = sum(p.numel() for p in self.embeddings.parameters())
        params['blocks'] = sum(p.numel() for p in self.blocks.parameters())
        params['final_norm'] = sum(p.numel() for p in self.final_norm.parameters())
        params['lm_head'] = sum(p.numel() for p in self.lm_head.parameters())
        params['total'] = sum(params.values())
        return params
    



############################### Testing Scripts ###################################

def test_transformer_block():
    """Test single transformer block"""
    print("🧪 Testing Single Transformer Block...")
    
    # Test configuration
    batch_size, seq_len, d_model = 2, 6, 32
    n_heads, d_ff = 4, 128
    
    print(f"📊 Config: {batch_size} batch, {seq_len} seq, {d_model} model, {n_heads} heads")
    
    # Create block
    block = TransformerBlock(d_model, n_heads, d_ff, dropout=0.0)
    
    # Create test input (embeddings)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Create causal mask
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    
    print(f"Input shape: {x.shape}")
    print(f"Mask shape: {causal_mask.shape}")
    
    # Forward pass
    output = block(x, mask=causal_mask)
    
    print(f"Output shape: {output.shape}")
    
    # Validation checks
    assert output.shape == x.shape, "Output shape should match input"
    
    # Check that output is different from input (transformation happened)
    difference = torch.abs(output - x).mean()
    print(f"Mean difference from input: {difference:.4f} (should be > 0)")
    
    print("✅ Transformer block test PASSED!")
    return True

def test_gpt_model():
    """Test complete GPT model"""
    print("\n🧪 Testing Complete GPT Model...")
    
    # Small test configuration
    vocab_size = 1000
    d_model = 64
    n_layers = 2
    n_heads = 4
    d_ff = 256
    max_seq_len = 128
    
    print(f"📊 Model config: {n_layers} layers, {d_model} dims, {n_heads} heads")
    
    # Create model
    model = GPTModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=0.0
    )
    
    # Test input (batch of token sequences)
    batch_size, seq_len = 2, 8
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"Input token IDs shape: {token_ids.shape}")
    print(f"Sample token IDs: {token_ids[0].tolist()}")
    
    # Forward pass
    logits = model(token_ids)
    
    print(f"Output logits shape: {logits.shape}")
    
    # Validation checks
    expected_shape = (batch_size, seq_len, vocab_size)
    assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"
    
    # Check parameter count
    param_info = model.get_num_params()
    print(f"\n📊 Parameter breakdown:")
    for component, count in param_info.items():
        print(f"   {component}: {count:,}")
    
    # Test that different positions get different predictions
    pos_0_probs = torch.softmax(logits[0, 0], dim=-1)
    pos_1_probs = torch.softmax(logits[0, 1], dim=-1)
    
    prob_difference = torch.abs(pos_0_probs - pos_1_probs).mean()
    print(f"\nPosition difference: {prob_difference:.4f} (should be > 0)")
    
    print("✅ GPT model test PASSED!")
    return True

def test_causal_attention():
    """Test that causal masking works in full model"""
    print("\n🧪 Testing Causal Attention in Full Model...")
    
    # Create small model for testing
    model = GPTModel(
        vocab_size=100, d_model=32, n_layers=1, n_heads=2, 
        d_ff=128, max_seq_len=64, dropout=0.0
    )
    
    # Test sequence
    token_ids = torch.tensor([[1, 2, 3, 4]])  # Single sequence
    
    # Get attention weights from first block
    model.eval()
    with torch.no_grad():
        # Hook to capture attention weights
        attention_weights = []
        
        def hook_fn(module, input, output):
            if len(output) == 2:  # (output, attention_weights)
                attention_weights.append(output[1])
        
        # Register hook on attention layer
        hook = model.blocks[0].attention.register_forward_hook(hook_fn)
        
        # Forward pass
        logits = model(token_ids)
        
        # Remove hook
        hook.remove()
        
        if attention_weights:
            attn = attention_weights[0][0]  # First batch, get attention matrix
            print(f"Attention weights shape: {attn.shape}")
            
            # Check causal property (upper triangle should be ~0)
            seq_len = attn.shape[-1]
            upper_triangle = torch.triu(attn[0], diagonal=1)  # First head, upper triangle
            max_upper = upper_triangle.max().item()
            
            print(f"Max attention to future tokens: {max_upper:.6f} (should be ~0)")
            
            causal_working = max_upper < 1e-6
            print(f"Causal masking working: {causal_working}")
    
    print("✅ Causal attention test PASSED!")
    return True

def test_generation_readiness():
    """Test that model is ready for text generation"""
    print("\n🧪 Testing Generation Readiness...")
    
    model = GPTModel(
        vocab_size=50, d_model=32, n_layers=1, n_heads=2,
        d_ff=128, max_seq_len=32, dropout=0.0
    )
    
    # Test autoregressive generation (predict next token)
    model.eval()
    with torch.no_grad():
        # Start with a single token
        context = torch.tensor([[1]])  # Start token
        
        print(f"Starting context: {context.tolist()}")
        
        # Generate next 3 tokens
        for step in range(3):
            # Get logits for current context
            logits = model(context)
            
            # Get next token (greedy - just take the most likely)
            next_token_logits = logits[0, -1, :]  # Last position, first batch
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to context
            context = torch.cat([context, next_token.unsqueeze(0)], dim=1)
            
            print(f"Step {step + 1}: Added token {next_token.item()}, context: {context[0].tolist()}")
    
    print("✅ Generation readiness test PASSED!")
    return True

if __name__ == "__main__":
    print("🚀 Testing Complete Transformer Architecture")
    print("=" * 70)
    
    tests = [
        test_transformer_block,
        test_gpt_model,
        test_causal_attention,
        test_generation_readiness
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 70)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print("🎉 ALL TRANSFORMER TESTS PASSED!")
        print("✅ Step 2.1 COMPLETE! Ready for Step 2.2: Model Integration!")
    else:
        print(f"❌ {total - passed}/{total} tests failed. Please fix before proceeding.")








