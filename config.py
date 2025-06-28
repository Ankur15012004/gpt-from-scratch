"""
Configuration file for LLM training
Contains all hyperparameters and settings in one place
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """
    Configuration for the Transformer model architecture
    
    Think of this as the "blueprint" for our model:
    - How big should it be?
    - How many layers?
    - How much attention?
    """

    # Vocabulary and embedding dimensions
    vocab_size: int = 50257        # GPT-2's vocabulary size (tiktoken)
    d_model: int = 768             # Hidden dimension 

    # Transformer architecture 
    n_layers: int = 12             # Number of transformer blocks (depth)
    n_heads: int = 12              # Number of attention heads (multi head attention)
    d_ff: int = 3072               # Feed-forward dimension (4*d_model is standard)

    # Sequence and context 
    block_size: int = 1024         # Maximum sequence length (context window)

    # Regularization 
    dropout: float = 0.1           # Dropout rate 

    def __post_init__(self):
        """
        Validate configuration after initialization
        
        This catches common mistakes before they cause errors later
        """

        # d_model must be divisible by n_heads for multi-head attention
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        
        # Calculate head dimension
        self.head_dim = self.d_model // self.n_heads
        print(f"✅ Model config: {self.n_layers} layers, {self.n_heads} heads, head_dim={self.head_dim}")


@dataclass
class TrainingConfig:
    """
    Configuration for training process
    
    Think of this as "how to teach the model":
    - How fast should it learn?
    - How much data at once?
    - When to save progress?
    """
    # Learning dynamics
    learning_rate: float = 3e-4    # How big steps to take (Adam optimizer default)
    min_learning_rate: float = 3e-5 # Minimum LR for scheduling

    # Batch processing
    batch_size: int = 12           # How many sequences to process together
    gradient_accumulation_steps: int = 5  # Simulate larger batches

    # Training duration
    max_epochs: int = 10           # How many times to see the entire dataset
    max_steps: Optional[int] = None # Alternative: train for N steps instead

    # Optimization
    weight_decay: float = 0.01     # L2 regularization strength
    beta1: float = 0.9             # Adam momentum parameter
    beta2: float = 0.95            # Adam momentum parameter (for transformers)

    # Checkpointing and logging
    save_every_steps: int = 1000   # Save model every N steps
    log_every_steps: int = 100     # Print progress every N steps
    eval_every_steps: int = 500    # Evaluate model every N steps

    # Hardware
    device: str = "cuda"           # Use GPU if available, else CPU
    compile_model: bool = False    # PyTorch 2.0 compilation (faster)

    def __post_init__(self):
        """Set device automatically if not specified"""
        import torch
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            print("⚠️  CUDA not available, using CPU")
        
        # Calculate effective batch size
        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps
        print(f"✅ Training config: effective batch size = {self.effective_batch_size}")

    
@dataclass
class DataConfig:
    """
    Configuration for data processing
    
    Think of this as "how to prepare the training material":
    - What text to use?
    - How to split it up?
    - How to load it efficiently?
    """

    # Data paths
    train_data_path: str = "data/sample.txt"  # Where is our training text?

    # Data processing 
    block_size: int = 1024         # FIXED: Added type annotation

    # Data loading 
    num_workers: int = 0
    pin_memory: bool = True

    def __post_init__(self):
        """Validate data configuration"""
        import os
        if not os.path.exists(self.train_data_path):
            print(f"⚠️  Training data not found at: {self.train_data_path}")
            print("   Please add training text to this file")


@dataclass
class Config:
    """Master configuration containing all sub-configurations"""
    model: ModelConfig = None
    training: TrainingConfig = None
    data: DataConfig = None

    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.data is None:
            self.data = DataConfig()

        # FIXED: Data should match model, not the other way around
        if self.model.block_size != self.data.block_size:
            self.data.block_size = self.model.block_size

    def print_config(self):
        """Print all configuration settings nicely"""
        print("\n" + "="*50)
        print("🔧 CONFIGURATION SUMMARY")
        print("="*50)
        print(f"\n📐 MODEL:")
        print(f"   • Vocabulary: {self.model.vocab_size:,} tokens")
        print(f"   • Model size: {self.model.d_model} dimensions")
        print(f"   • Architecture: {self.model.n_layers} layers x {self.model.n_heads} heads")
        print(f"   • Context: {self.model.block_size} tokens")

        print(f"\n🎯 TRAINING:")
        print(f"   • Learning rate: {self.training.learning_rate}")
        print(f"   • Effective batch size: {self.training.effective_batch_size}")
        print(f"   • Device: {self.training.device}")
        print(f"   • Max epochs: {self.training.max_epochs}")

        print(f"\n📚 DATA:")
        print(f"   • Training file: {self.data.train_data_path}")
        print(f"   • Block size: {self.data.block_size}")
        print("="*50 + "\n")

def get_default_config() -> Config:
    """Get default configuration for quick start"""
    return Config()

def get_small_config() -> Config:
    """Get smaller configuration for testing/development"""
    model_config = ModelConfig(
        d_model=128, n_layers=4, n_heads=4, block_size=256, d_ff=512
    )
    training_config = TrainingConfig(
        batch_size=4,gradient_accumulation_steps=8, max_epochs=3, learning_rate=1e-3
    )
    return Config(model=model_config, training=training_config)

def test_config():
    """Test our configuration system"""
    print("🧪 Testing Configuration System...")
    print("\n1️⃣ Default Configuration:")
    default_config = get_default_config()
    default_config.print_config()
    
    print("\n2️⃣ Small Configuration:")
    small_config = get_small_config()
    small_config.print_config()
    
    print("✅ Configuration system working!")

if __name__ == "__main__":
    test_config()
