"""
Interactive Text Generation with Trained GPT Model
Usage: python generate.py --prompt "HAMLET:" --length 100 --temperature 0.8
"""

import argparse
import torch
import os
from datetime import datetime

from config import get_small_config
from src.data.tokenizer import GPTTokenizer
from src.model.transformer import GPTModel

def load_trained_model(checkpoint_path="checkpoints/best_model.pt"):
    """Load the trained model from checkpoint"""
    print(f"🔄 Loading trained model from: {checkpoint_path}")
    
    # Load configuration
    config = get_small_config()
    
    # Create model architecture
    model = GPTModel(
        vocab_size=config.model.vocab_size,
        d_model=config.model.d_model,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        d_ff=config.model.d_ff,
        max_seq_len=config.model.block_size,
        dropout=0.0  # No dropout during inference
    )
    
    # Load trained weights
    if os.path.exists(checkpoint_path):
        if 'latest.pt' in checkpoint_path or 'emergency' in checkpoint_path:
            # Full checkpoint format
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded full checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            # Just model weights
            model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
            print(f"✅ Loaded model weights")
    else:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Available checkpoints:")
        if os.path.exists("checkpoints"):
            for f in os.listdir("checkpoints"):
                print(f"  - checkpoints/{f}")
        return None, None
    
    # Create tokenizer
    tokenizer = GPTTokenizer()
    
    model.eval()  # Set to evaluation mode
    print(f"🎭 Model ready for text generation!")
    
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8, top_k=None):
    """Generate text from prompt with various sampling strategies"""
    
    print(f"\n🎯 Generating text...")
    print(f"📝 Prompt: '{prompt}'")
    print(f"🌡️  Temperature: {temperature}")
    print(f"📏 Max length: {max_length} tokens")
    
    # Encode prompt
    token_ids = tokenizer.encode(prompt)
    if len(token_ids) == 0:
        token_ids = [0]  # Fallback for empty prompt
    
    context = torch.tensor([token_ids])
    generated_tokens = token_ids.copy()
    
    with torch.no_grad():
        for step in range(max_length):
            # Forward pass
            logits = model(context)
            next_token_logits = logits[0, -1, :]  # Last position logits
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering if specified
            if top_k is not None:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                filtered_logits = torch.full_like(next_token_logits, float('-inf'))
                filtered_logits[top_k_indices] = top_k_logits
                next_token_logits = filtered_logits
            
            # Sample next token
            if temperature <= 0.1:
                # Greedy sampling
                next_token = torch.argmax(next_token_logits, dim=-1).item()
            else:
                # Temperature sampling
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            
            # Add to sequence
            generated_tokens.append(next_token)
            context = torch.cat([context, torch.tensor([[next_token]])], dim=1)
            
            # Stop if we hit context limit
            if context.shape[1] >= model.max_seq_len:
                print(f"⚠️  Reached max context length ({model.max_seq_len})")
                break
    
    # Decode generated text
    try:
        generated_text = tokenizer.decode(generated_tokens)
        print(f"\n📖 Generated Text:")
        print(f"{'='*60}")
        print(generated_text)
        print(f"{'='*60}")
        print(f"📊 Generated {len(generated_tokens)} tokens in {step + 1} steps")
        
        return generated_text, generated_tokens
        
    except Exception as e:
        print(f"❌ Decoding failed: {e}")
        return None, generated_tokens

def interactive_mode(model, tokenizer):
    """Interactive text generation loop"""
    print(f"\n🎭 Interactive Shakespeare Generation Mode")
    print(f"{'='*60}")
    print(f"💡 Try prompts like:")
    print(f"   - 'HAMLET:'")
    print(f"   - 'First Citizen:'") 
    print(f"   - 'To be or'")
    print(f"   - 'Once upon a time'")
    print(f"{'='*60}")
    print(f"🔧 Commands: 'quit' to exit, 'save' to save last generation")
    print(f"{'='*60}")
    
    last_generation = ""
    
    while True:
        try:
            # Get user input
            prompt = input(f"\n🎯 Enter prompt (or 'quit'): ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print(f"👋 Goodbye! Thanks for using the Shakespeare Generator!")
                break
            
            if prompt.lower() == 'save' and last_generation:
                # Save last generation
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"generated_text_{timestamp}.txt"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"Generated Text - {datetime.now()}\n")
                    f.write(f"{'='*60}\n")
                    f.write(last_generation)
                print(f"💾 Saved to: {filename}")
                continue
            
            if not prompt:
                print(f"❌ Please enter a prompt!")
                continue
            
            # Get generation parameters
            try:
                temp_input = input(f"🌡️  Temperature (0.1-2.0, default 0.8): ").strip()
                temperature = float(temp_input) if temp_input else 0.8
                temperature = max(0.1, min(2.0, temperature))  # Clamp
                
                len_input = input(f"📏 Max length (10-200, default 50): ").strip()
                max_length = int(len_input) if len_input else 50
                max_length = max(10, min(200, max_length))  # Clamp
                
            except ValueError:
                print(f"⚠️  Using default parameters (temp=0.8, length=50)")
                temperature = 0.8
                max_length = 50
            
            # Generate text
            generated_text, _ = generate_text(
                model, tokenizer, prompt, 
                max_length=max_length,
                temperature=temperature
            )
            
            if generated_text:
                last_generation = generated_text
            
        except KeyboardInterrupt:
            print(f"\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="Generate text with trained GPT model")
    parser.add_argument("--prompt", type=str, help="Text prompt for generation")
    parser.add_argument("--length", type=int, default=100, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, help="Top-k sampling (optional)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt", 
                       help="Path to model checkpoint")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    print(f"🎭 Shakespeare Text Generator")
    print(f"Built with Transformer from Scratch")
    print(f"{'='*60}")
    
    # Load model
    model, tokenizer = load_trained_model(args.checkpoint)
    if model is None:
        return
    
    if args.interactive or not args.prompt:
        # Interactive mode
        interactive_mode(model, tokenizer)
    else:
        # Single generation mode
        generated_text, _ = generate_text(
            model, tokenizer, args.prompt,
            max_length=args.length,
            temperature=args.temperature,
            top_k=args.top_k
        )

if __name__ == "__main__":
    main()
