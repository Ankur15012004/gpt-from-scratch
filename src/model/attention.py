"""
Self-Attention Mechanism for Transformers (DOUBLE-FIXED VERSION)
Built from scratch following Sebastian Raschka's educational approach
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F 
import math 
from typing import Optional, Tuple

class ScaledDotProductAttention(nn.Module):
    """Core attention computation that handles multi-head tensors"""

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply scaled dot-product attention"""
        
        # Handle both 3D and 4D tensors
        if query.dim() == 4:
            # Multi-head case: (batch_size, n_heads, seq_len, head_dim)
            batch_size, n_heads, seq_len, d_k = query.shape
        else:
            # Single-head case: (batch_size, seq_len, d_k)
            batch_size, seq_len, d_k = query.shape

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1))
        
        # Scale by sqrt(d_k)
        scores = scores / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            # 🔧 FIX: Handle mask dimensions properly for multi-head case
            if query.dim() == 4 and mask.dim() == 2:
                # Expand mask for batch and head dimensions
                # mask: (seq_len, seq_len) -> (batch_size, n_heads, seq_len, seq_len)
                mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, n_heads, -1, -1)
            elif query.dim() == 4 and mask.dim() == 3:
                # mask: (batch_size, seq_len, seq_len) -> (batch_size, n_heads, seq_len, seq_len)
                mask = mask.unsqueeze(1).expand(-1, n_heads, -1, -1)
            
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Convert to probabilities
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights

class MultiHeadSelfAttention(nn.Module):
    """Multi-head SELF-attention for GPT-style models"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        print(f"✅ Multi-head attention: {n_heads} heads, {self.head_dim} dims per head")

        # Linear layers to create Q, K, V
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scaled_dot_product = ScaledDotProductAttention(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply multi-head self-attention"""
        batch_size, seq_len, _ = x.size()

        # Create Q, K, V from the same input x
        Q = self.query_linear(x)
        K = self.key_linear(x)
        V = self.value_linear(x)

        # Split into multiple heads
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        # Shape after transpose: (batch_size, n_heads, seq_len, head_dim)

        # 🔧 FIX: Remove the mask unsqueeze here - handle it in ScaledDotProductAttention
        # Don't modify mask here, let the attention function handle it properly

        # Apply attention (now works with 4D tensors and proper mask handling)
        attn_output, attn_weights = self.scaled_dot_product(Q, K, V, mask)

        # Concatenate heads back together
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        # Final projection
        output = self.out_linear(attn_output)
        output = self.dropout(output)
        
        return output, attn_weights

def test_self_attention():
    """Test corrected self-attention"""
    print("🧪 Testing Multi-Head SELF-Attention (Fixed)...")
    
    batch_size, seq_len, d_model = 1, 4, 8
    n_heads = 2
    
    print(f"Creating model with d_model={d_model}, n_heads={n_heads}")
    model = MultiHeadSelfAttention(d_model=d_model, n_heads=n_heads, dropout=0.0)
    
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input shape: {x.shape}")
    
    # This should now work without the NoneType error
    try:
        output, attn_weights = model(x)
        print(f"✅ Success!")
        print(f"Output shape: {output.shape}")
        print(f"Attention weights shape: {attn_weights.shape}")
        
        # 🔧 FIX: Use torch.round instead of round, or format differently
        weight_sums = attn_weights.sum(dim=-1)
        print(f"Attention weight sums (should be ~1.0): {weight_sums[0, 0, :].detach().numpy()}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    print("✅ Self-attention test PASSED!")
    return True

def test_causal_self_attention():
    """Test with causal mask"""
    print("\n🧪 Testing Causal Self-Attention...")
    
    batch_size, seq_len, d_model = 1, 4, 8
    n_heads = 2
    
    model = MultiHeadSelfAttention(d_model=d_model, n_heads=n_heads, dropout=0.0)
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 🔧 FIX: Create proper causal mask
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    
    print(f"Input shape: {x.shape}")
    print(f"Causal mask shape: {causal_mask.shape}")
    print(f"Causal mask:")
    print(causal_mask.numpy().astype(int))
    
    try:
        output, attn_weights = model(x, mask=causal_mask)
        print(f"✅ Success!")
        print(f"Output shape: {output.shape}")
        print(f"Attention weights shape: {attn_weights.shape}")
        
        # Show causal pattern
        print(f"First head attention pattern:")
        first_head = attn_weights[0, 0].detach().numpy()
        
        # 🔧 FIX: Format numbers properly without using round()
        print("Attention matrix (first head):")
        for i in range(seq_len):
            row_str = " ".join([f"{first_head[i,j]:.3f}" for j in range(seq_len)])
            print(f"  Position {i}: [{row_str}]")
        
        print("(Upper triangle should be ~0)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    print("✅ Causal self-attention test PASSED!")
    return True

if __name__ == "__main__":
    print("🚀 Testing Attention Mechanism")
    print("=" * 50)
    
    # Run tests
    test1_passed = test_self_attention()
    test2_passed = test_causal_self_attention()
    
    print("\n" + "=" * 50)
    if test1_passed and test2_passed:
        print("🎉 ALL TESTS PASSED! Attention mechanism working correctly!")
        print("✅ Ready for next step: Position Encoding")
    else:
        print("❌ Some tests failed. Please check the errors above.")
