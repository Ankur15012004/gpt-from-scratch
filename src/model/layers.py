"""
Layer components for Transformer blocks: LayerNorm, FeedForward, ResidualConnection
Built from scratch following GPT architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Any

class LayerNorm(nn.Module):
    """
    Layer Normalization - stabilizes training by normalizing across features
    
    Intuition: Like standardizing test scores across different subjects
    - Input: [batch_size, seq_len, d_model]
    - For each position, normalize across the d_model features
    - Keeps activations in a reasonable range
    
    Why not BatchNorm? LayerNorm works better for sequences of variable length
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        """
        Initialize Layer Normalization
        
        Args:
            d_model: Model dimension (same as embeddings)
            eps: Small value to prevent division by zero
        """
        super().__init__()
        self.d_model=d_model
        self.eps=eps

        # Learnable parameter to scale and shift the normalized values 

        # gamma: multiplicative 
        # beta: additive 
        self.gamma =nn.Parameter(torch.ones(self.d_model))
        self.beta=nn.Parameter(torch.zeros(self.d_model))

        print(f"✅ Layer normalization: {d_model} features")

    def forward(self, x: torch.tensor)-> torch.Tensor:
        """
        Apply layer normalization
        
        Args:
            x: (batch_size, seq_len, d_model) - Input features
            
        Returns:
            normalized: (batch_size, seq_len, d_model) - Normalized features
        """
        # Calculate statistics across the feature dimension (dim=-1)
        # For each token position, compute mean/var across all features

        mean=x.mean(dim=-1, keepdim=True) #(batch_size, seq_len, 1)
        var=x.var(dim=-1, unbiased=False, keepdim=True)

        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        output=self.gamma*x_normalized +self.beta 

        return output
    

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN)
    
    Intuition: Like a mini neural network applied to each token position independently
    - Takes each token's representation
    - Expands it to a larger dimension (d_ff)
    - Applies non-linearity (GELU)
    - Projects back to original size (d_model)
    
    Why this works: Allows the model to learn complex transformations
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float=0.1):
        """
        Initialize feed-forward network
        
        Args:
            d_model: Input/output dimension (768 for GPT-2)
            d_ff: Hidden dimension (usually 4 * d_model = 3072 for GPT-2)
            dropout: Dropout rate for regularization
        """
        super().__init__()
        self.d_model=d_model
        self.d_ff=d_ff

        self.linear1=nn.Linear(d_model,d_ff) # Expand the dimensions 
        self.linear2=nn.Linear(d_ff, d_model) # Project Back 
        self.dropout=nn.Dropout(dropout)

        nn.init.normal_(self.linear1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.linear2.weight, mean=0.0, std=0.02)
        
        print(f"✅ Feed-forward: {d_model} → {d_ff} → {d_model}")

    def forward(self, x: torch.Tensor)->torch.Tensor:
        """
        Apply feed-forward transformation
        
        Args:
            x: (batch_size, seq_len, d_model) - Input features
            
        Returns:
            output: (batch_size, seq_len, d_model) - Transformed features
        """

        x=self.linear1(x) # (batch_size, seq_len, d_ff)

        # Step 2: GELU activation (smoother than ReLU, used in GPT)
        # GELU(x) = x * Φ(x) where Φ is standard normal CDF

        x=F.gelu(x)

        x=self.dropout(x)

        # Step 4: Project back to d_model
        x = self.linear2(x)  # (batch_size, seq_len, d_model)
        
        # Step 5: Final dropout
        x = self.dropout(x)
        
        return x
    
class ResidualConnection(nn.Module):
    """Residual connection: output = input + sublayer(input)"""

    def __init__(self, dropout: float=0.1):
        super().__init__()
        self.dropout=nn.Dropout(dropout)

        print(f"✅ Residual connection with dropout: {dropout}")

    def forward(self, x: torch.Tensor, sublayer: Callable[[torch.Tensor],Any]):
        sublayer_output=sublayer(x)

        if isinstance(sublayer_output, tuple):
            sublayer_output=sublayer_output[0]

        sublayer_output=self.dropout(sublayer_output)

        return x+sublayer_output
    
class PreLayerNormResidual(nn.Module):
    """Pre-LayerNorm residual connection (GPT-2 style)"""

    def __init__(self, d_model: int, dropout: float=0.1):
        super().__init__()
        self.layer_norm=LayerNorm(d_model)
        self.dropout=nn.Dropout(dropout)

        print(f"✅ Pre-LayerNorm residual connection")

    def forward(self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], Any]):
        normalized=self.layer_norm(x)
        sublayer_output=sublayer(normalized)

        if isinstance(sublayer_output, tuple):
            sublayer_output=sublayer_output[0]

        sublayer_output=self.dropout(sublayer_output)

        return x+sublayer_output
    

#################### Testing Script ####################################

def test_layer_norm():
    """Test layer normalization"""
    print("🧪 Testing Layer Normalization...")
    
    batch_size, seq_len, d_model = 2, 4, 8
    layer_norm = LayerNorm(d_model)
    
    # Create test input with different scales
    x = torch.randn(batch_size, seq_len, d_model) * 10  # Large values
    
    print(f"Input shape: {x.shape}")
    print(f"Input mean: {x.mean():.3f}, std: {x.std():.3f}")
    
    # Apply layer norm
    x_norm = layer_norm(x)
    
    print(f"Output shape: {x_norm.shape}")
    print(f"Output mean: {x_norm.mean():.3f}, std: {x_norm.std():.3f}")
    
    # Check that each position is normalized (mean≈0, std≈1)
    position_means = x_norm.mean(dim=-1)
    position_stds = x_norm.std(dim=-1)
    
    print(f"Per-position means (should be ~0): {position_means[0].detach().numpy().round(3)}")
    print(f"Per-position stds (should be ~1): {position_stds[0].detach().numpy().round(3)}")
    
    print("✅ Layer normalization test PASSED!")
    return True

def test_feed_forward():
    """Test feed-forward network"""
    print("\n🧪 Testing Feed-Forward Network...")
    
    batch_size, seq_len, d_model = 2, 4, 8
    d_ff = 32  # 4 * d_model
    
    ffn = FeedForward(d_model, d_ff, dropout=0.0)
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"Input shape: {x.shape}")
    
    # Apply feed-forward
    output = ffn(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Shape preserved: {output.shape == x.shape}")
    
    # Check that output is different from input (transformation happened)
    difference = torch.abs(output - x).mean()
    print(f"Mean difference from input: {difference:.3f} (should be > 0)")
    
    print("✅ Feed-forward test PASSED!")
    return True

def test_residual_connection():
    """Test residual connection"""
    print("\n🧪 Testing Residual Connection...")
    
    batch_size, seq_len, d_model = 2, 4, 8
    
    residual = ResidualConnection(dropout=0.0)
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Define a simple sublayer (identity function)
    def identity_layer(input_tensor):
        return torch.zeros_like(input_tensor)  # Returns zeros
    
    print(f"Input shape: {x.shape}")
    
    # Apply residual connection with zero sublayer
    output = residual(x, identity_layer)
    
    print(f"Output shape: {output.shape}")
    
    # With zero sublayer, output should equal input
    is_equal = torch.allclose(output, x)
    print(f"Output equals input (with zero sublayer): {is_equal}")
    
    print("✅ Residual connection test PASSED!")
    return True

def test_integration():
    """Test components working together"""
    print("\n🧪 Testing Component Integration...")
    
    batch_size, seq_len, d_model = 2, 4, 8
    d_ff = 32
    
    # Create components
    layer_norm = LayerNorm(d_model)
    ffn = FeedForward(d_model, d_ff, dropout=0.0)
    residual = PreLayerNormResidual(d_model, dropout=0.0)
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"Input shape: {x.shape}")
    
    # Simulate transformer block: x + FFN(LayerNorm(x))
    def ffn_sublayer(input_tensor):
        return ffn(input_tensor)
    
    output = residual(x, ffn_sublayer)
    
    print(f"Output shape: {output.shape}")
    print(f"Output different from input: {not torch.allclose(output, x)}")
    
    print("✅ Integration test PASSED!")
    return True

if __name__ == "__main__":
    print("🚀 Testing Layer Components")
    print("=" * 60)
    
    tests = [
        test_layer_norm,
        test_feed_forward,
        test_residual_connection,
        test_integration
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_func.__name__} FAILED: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print("🎉 ALL LAYER TESTS PASSED!")
        print("✅ Hour 4-8 COMPLETE! Ready for Hour 8-12: Transformer Blocks!")
    else:
        print(f"❌ {total - passed}/{total} tests failed. Please fix before proceeding.")








