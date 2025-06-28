"""
Main training script for GPT model from scratch
Complete training pipeline with progress tracking, checkpointing, and text generation
"""

import os
import torch
import time
from datetime import datetime
from torch.utils.data import DataLoader
import math

# Import our components
from config import get_small_config
from src.data.tokenizer import GPTTokenizer
from src.data.dataset import TextDataset
from src.model.transformer import GPTModel
from src.utils.training import Trainer, create_training_pipeline

def generate_text(model, tokenizer, prompt: str, max_length: int = 100, temperature: float = 1.0):
    """
    Generate text from a prompt with temperature sampling
    
    Args:
        model: Trained GPT model
        tokenizer: Tokenizer for encoding/decoding
        prompt: Starting text prompt
        max_length: Maximum tokens to generate
        temperature: Sampling temperature (1.0 = normal, <1.0 = more focused)
    
    Returns:
        generated_text: Complete generated text including prompt
        generated_tokens: List of all token IDs
    """
    model.eval()
    
    # Encode prompt
    token_ids = tokenizer.encode(prompt)
    if len(token_ids) == 0:
        token_ids = [0]  # Use a default token if prompt is empty
    
    context = torch.tensor([token_ids])
    generated_tokens = token_ids.copy()
    
    print(f"🎯 Generating from prompt: '{prompt}'")
    print(f"📝 Starting with {len(token_ids)} tokens")
    
    with torch.no_grad():
        for step in range(max_length):
            # Forward pass
            logits = model(context)
            next_token_logits = logits[0, -1, :] / temperature  # Apply temperature
            
            # Sample next token (greedy if temperature=1.0)
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
                print(f"⚠️  Reached max sequence length ({model.max_seq_len})")
                break
    
    # Decode generated text
    try:
        generated_text = tokenizer.decode(generated_tokens)
    except:
        # Fallback if decoding fails
        generated_text = f"[Generated {len(generated_tokens)} tokens]"
    
    return generated_text, generated_tokens

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    """Save model checkpoint with metadata"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat(),
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f"gpt_epoch_{epoch:03d}.pt")
    torch.save(checkpoint, checkpoint_path)
    
    # Also save as latest
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    torch.save(checkpoint, latest_path)
    
    return checkpoint_path

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss']

def main():
    """Enhanced main training function with safe interrupt handling"""
    print("🚀 Starting GPT Training from Scratch")
    print("=" * 60)
    
    # Load configuration
    config = get_small_config()
    print(f"📊 Configuration loaded:")
    print(f"   Model: {config.model.n_layers} layers, {config.model.d_model} dims, {config.model.n_heads} heads")
    print(f"   Training: {config.training.max_epochs} epochs, batch size {config.training.batch_size}")
    print(f"   Device: {config.training.device}")
    
    # Create training pipeline
    print(f"\n🔧 Setting up training pipeline...")
    model, trainer, dataloader, tokenizer = create_training_pipeline(config)
    
    # Create optimizer for checkpointing
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=(config.training.beta1, config.training.beta2)
    )
    
    # Setup directories
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training info
    num_epochs = config.training.max_epochs
    dataset_size = len(dataloader.dataset)
    batches_per_epoch = len(dataloader)
    
    print(f"\n📈 Training setup:")
    print(f"   Dataset: {dataset_size} sequences")
    print(f"   Batches per epoch: {batches_per_epoch}")
    print(f"   Total training steps: {num_epochs * batches_per_epoch}")
    print(f"   Model parameters: {trainer._count_parameters():,}")
    
    # Training loop with interrupt handling
    print(f"\n🎯 Starting training...")
    print(f"💡 Press Ctrl+C anytime to safely stop and save progress")
    start_time = time.time()
    best_loss = float('inf')
    
    try:
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            print(f"\n" + "="*40)
            print(f"📚 EPOCH {epoch + 1}/{num_epochs}")
            print(f"="*40)
            
            # Enhanced epoch training with interrupt handling
            try:
                epoch_metrics = train_epoch_with_interrupt_handling(
                    trainer, dataloader, epoch, checkpoint_dir, 
                    model, optimizer, batches_per_epoch
                )
            except KeyboardInterrupt:
                # Save emergency checkpoint mid-epoch
                print(f"\n🛑 Training interrupted during epoch {epoch + 1}")
                emergency_checkpoint = save_emergency_checkpoint(
                    model, optimizer, epoch, trainer.train_losses[-1] if trainer.train_losses else float('inf'), 
                    checkpoint_dir, step=trainer.step_count
                )
                print(f"💾 Emergency checkpoint saved: {emergency_checkpoint}")
                raise  # Re-raise to trigger main handler
            
            # Calculate timing
            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time
            
            # Log results
            train_loss = epoch_metrics['train_loss']
            train_ppl = epoch_metrics['train_perplexity']
            
            print(f"\n📊 Epoch {epoch + 1} Results:")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Train Perplexity: {train_ppl:.2f}")
            print(f"   Epoch Time: {epoch_time:.1f}s")
            print(f"   Total Time: {total_time/60:.1f}m")
            
            # Save regular checkpoint
            checkpoint_path = save_checkpoint(model, optimizer, epoch + 1, train_loss, checkpoint_dir)
            print(f"💾 Saved checkpoint: {checkpoint_path}")
            
            # Track best model
            if train_loss < best_loss:
                best_loss = train_loss
                best_path = os.path.join(checkpoint_dir, "best_model.pt")
                torch.save(model.state_dict(), best_path)
                print(f"🏆 New best model saved! Loss: {best_loss:.4f}")
            
            # Generate sample text
            print(f"\n🎭 Sample Generation:")
            sample_prompts = ["First Citizen:", "HAMLET:", "To be or"]
            
            for prompt in sample_prompts:
                try:
                    generated_text, _ = generate_text(
                        model, tokenizer, prompt, 
                        max_length=30, temperature=0.8
                    )
                    display_text = generated_text[:100] + "..." if len(generated_text) > 100 else generated_text
                    print(f"   '{prompt}' → '{display_text}'")
                except Exception as e:
                    print(f"   '{prompt}' → [Generation failed: {e}]")
        
        # Training complete normally
        total_time = time.time() - start_time
        print(f"\n" + "="*60)
        print(f"🎉 TRAINING COMPLETE!")
        print(f"   Total time: {total_time/60:.1f} minutes")
        print(f"   Best loss: {best_loss:.4f}")
        print(f"   Final loss: {train_loss:.4f}")
        print(f"   Checkpoints saved in: {checkpoint_dir}")
        print(f"="*60)
        
    except KeyboardInterrupt:
        # This catches interrupts from the main loop
        total_time = time.time() - start_time
        print(f"\n🛑 TRAINING SAFELY INTERRUPTED")
        print(f"   Training time: {total_time/60:.1f} minutes")
        print(f"   Progress saved in: {checkpoint_dir}")
        print(f"   Best loss so far: {best_loss:.4f}")
        raise  # Re-raise to trigger final handler
    
    return model, trainer, tokenizer

def train_epoch_with_interrupt_handling(trainer, dataloader, epoch, checkpoint_dir, model, optimizer, batches_per_epoch):
    """Enhanced epoch training with mid-epoch checkpoint saving"""
    from tqdm import tqdm
    
    epoch_losses = []
    
    # Progress bar with interrupt-friendly settings
    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}", leave=True)
    
    try:
        for batch_idx, batch in enumerate(pbar):
            # Training step
            metrics = trainer.train_step(batch)
            epoch_losses.append(metrics['loss'])
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'ppl': f"{metrics['perplexity']:.2f}",
                'step': trainer.step_count
            })
            
            # Save intermediate checkpoint every 100 steps
            if trainer.step_count % 100 == 0:
                try:
                    intermediate_checkpoint = save_intermediate_checkpoint(
                        model, optimizer, epoch, metrics['loss'], 
                        checkpoint_dir, trainer.step_count, batch_idx, batches_per_epoch
                    )
                    # Don't print this every time, just save silently
                except Exception as e:
                    print(f"⚠️ Warning: Could not save intermediate checkpoint: {e}")
    
    except KeyboardInterrupt:
        # Save progress when interrupted mid-epoch
        pbar.close()
        print(f"\n🛑 Interrupted at batch {batch_idx + 1}/{batches_per_epoch}")
        
        # Save current progress
        if epoch_losses:
            current_loss = sum(epoch_losses) / len(epoch_losses)
        else:
            current_loss = float('inf')
        
        emergency_checkpoint = save_emergency_checkpoint(
            model, optimizer, epoch, current_loss, 
            checkpoint_dir, step=trainer.step_count, batch=batch_idx
        )
        print(f"💾 Progress saved: {emergency_checkpoint}")
        raise
    
    # Calculate epoch metrics
    avg_loss = sum(epoch_losses) / len(epoch_losses)
    avg_perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')
    
    trainer.epoch_count += 1
    trainer.train_losses.append(avg_loss)
    
    return {
        'epoch': trainer.epoch_count,
        'train_loss': avg_loss,
        'train_perplexity': avg_perplexity
    }

def save_intermediate_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, step, batch, total_batches):
    """Save intermediate checkpoint during training"""
    checkpoint = {
        'epoch': epoch,
        'batch': batch,
        'total_batches': total_batches,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat(),
        'status': 'intermediate'
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f"intermediate_step_{step:06d}.pt")
    torch.save(checkpoint, checkpoint_path)
    
    # Also update latest
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    torch.save(checkpoint, latest_path)
    
    return checkpoint_path

def save_emergency_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, step=0, batch=0):
    """Save emergency checkpoint when interrupted"""
    checkpoint = {
        'epoch': epoch,
        'batch': batch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat(),
        'status': 'interrupted'
    }
    
    emergency_path = os.path.join(checkpoint_dir, f"emergency_epoch_{epoch}_step_{step}.pt")
    torch.save(checkpoint, emergency_path)
    
    # Update latest as well
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    torch.save(checkpoint, latest_path)
    
    return emergency_path

def enhanced_resume_training():
    """Enhanced resume function that actually continues training"""
    print("🔄 Resuming Training from Checkpoint")
    print("=" * 50)
    
    checkpoint_dir = "checkpoints"
    latest_checkpoint = os.path.join(checkpoint_dir, "latest.pt")
    
    if not os.path.exists(latest_checkpoint):
        print(f"❌ No checkpoint found")
        return main()
    
    # Load and analyze checkpoint
    checkpoint = torch.load(latest_checkpoint, map_location='cpu')
    status = checkpoint.get('status', 'normal')
    start_epoch = checkpoint['epoch']
    last_loss = checkpoint['loss']
    
    print(f"📂 Found checkpoint: {status}")
    print(f"   Epoch: {start_epoch}")
    print(f"   Step: {checkpoint.get('step', 0)}")
    print(f"   Loss: {last_loss:.4f}" if last_loss != float('inf') else f"   Loss: {last_loss}")
    print(f"   Saved: {checkpoint.get('timestamp', 'unknown')}")
    
    if status == 'interrupted':
        print(f"🔄 Resuming from interrupted training...")
        batch_info = f" (batch {checkpoint.get('batch', 0)})" if 'batch' in checkpoint else ""
        print(f"   Continuing from epoch {start_epoch}{batch_info}")
    
    # Load configuration
    config = get_small_config()
    
    # Create training pipeline
    model, trainer, dataloader, tokenizer = create_training_pipeline(config)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=(config.training.beta1, config.training.beta2)
    )
    
    # Load checkpoint state
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Update trainer state
    trainer.epoch_count = start_epoch
    trainer.step_count = checkpoint.get('step', 0)
    if last_loss != float('inf'):
        trainer.train_losses = [last_loss]
    
    print(f"✅ Successfully resumed from checkpoint")
    
    # Calculate remaining epochs
    num_epochs = config.training.max_epochs
    remaining_epochs = num_epochs - start_epoch
    
    if remaining_epochs <= 0:
        print(f"🎉 Training was already complete! ({start_epoch}/{num_epochs} epochs)")
        return model, trainer, tokenizer
    
    print(f"🎯 Continuing training...")
    print(f"📈 {remaining_epochs} epochs remaining")
    
    # *** THIS IS THE MISSING PART - ACTUALLY CONTINUE TRAINING ***
    dataset_size = len(dataloader.dataset)
    batches_per_epoch = len(dataloader)
    
    print(f"\n📈 Resumed training setup:")
    print(f"   Dataset: {dataset_size} sequences")
    print(f"   Batches per epoch: {batches_per_epoch}")
    print(f"   Remaining epochs: {remaining_epochs}")
    print(f"   Model parameters: {trainer._count_parameters():,}")
    
    # CONTINUE THE ACTUAL TRAINING LOOP
    start_time = time.time()
    best_loss = last_loss if last_loss != float('inf') else float('inf')
    
    try:
        for epoch in range(start_epoch, num_epochs):
            epoch_start = time.time()
            
            print(f"\n" + "="*40)
            print(f"📚 EPOCH {epoch + 1}/{num_epochs} (RESUMED)")
            print(f"="*40)
            
            # Enhanced epoch training with interrupt handling
            try:
                epoch_metrics = train_epoch_with_interrupt_handling(
                    trainer, dataloader, epoch, checkpoint_dir, 
                    model, optimizer, batches_per_epoch
                )
            except KeyboardInterrupt:
                # Save emergency checkpoint mid-epoch
                print(f"\n🛑 Training interrupted during epoch {epoch + 1}")
                emergency_checkpoint = save_emergency_checkpoint(
                    model, optimizer, epoch, trainer.train_losses[-1] if trainer.train_losses else float('inf'), 
                    checkpoint_dir, step=trainer.step_count
                )
                print(f"💾 Emergency checkpoint saved: {emergency_checkpoint}")
                raise  # Re-raise to trigger main handler
            
            # Calculate timing
            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time
            
            # Log results
            train_loss = epoch_metrics['train_loss']
            train_ppl = epoch_metrics['train_perplexity']
            
            print(f"\n📊 Epoch {epoch + 1} Results:")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Train Perplexity: {train_ppl:.2f}")
            print(f"   Epoch Time: {epoch_time:.1f}s")
            print(f"   Total Time: {total_time/60:.1f}m")
            
            # Save regular checkpoint
            checkpoint_path = save_checkpoint(model, optimizer, epoch + 1, train_loss, checkpoint_dir)
            print(f"💾 Saved checkpoint: {checkpoint_path}")
            
            # Track best model
            if train_loss < best_loss:
                best_loss = train_loss
                best_path = os.path.join(checkpoint_dir, "best_model.pt")
                torch.save(model.state_dict(), best_path)
                print(f"🏆 New best model saved! Loss: {best_loss:.4f}")
            
            # Generate sample text
            print(f"\n🎭 Sample Generation:")
            sample_prompts = ["First Citizen:", "HAMLET:", "To be or"]
            
            for prompt in sample_prompts:
                try:
                    generated_text, _ = generate_text(
                        model, tokenizer, prompt, 
                        max_length=30, temperature=0.8
                    )
                    display_text = generated_text[:100] + "..." if len(generated_text) > 100 else generated_text
                    print(f"   '{prompt}' → '{display_text}'")
                except Exception as e:
                    print(f"   '{prompt}' → [Generation failed: {e}]")
        
        # Training complete
        total_time = time.time() - start_time
        print(f"\n" + "="*60)
        print(f"🎉 RESUMED TRAINING COMPLETE!")
        print(f"   Resume time: {total_time/60:.1f} minutes")
        print(f"   Best loss: {best_loss:.4f}")
        print(f"   Final loss: {train_loss:.4f}")
        print(f"   Checkpoints saved in: {checkpoint_dir}")
        print(f"="*60)
        
    except KeyboardInterrupt:
        # Handle interrupts during resumed training
        total_time = time.time() - start_time
        print(f"\n🛑 RESUMED TRAINING SAFELY INTERRUPTED")
        print(f"   Resume time: {total_time/60:.1f} minutes")
        print(f"   Progress saved in: {checkpoint_dir}")
        print(f"   Best loss so far: {best_loss:.4f}")
        raise  # Re-raise to trigger final handler
    
    return model, trainer, tokenizer



if __name__ == "__main__":
    import sys
    
    try:
        # Check for resume flag
        if len(sys.argv) > 1 and sys.argv[1] == "--resume":
            model, trainer, tokenizer = enhanced_resume_training()  # This will now actually train
        else:
            model, trainer, tokenizer = main()
            
        print(f"\n🚀 Training completed successfully!")
        print(f"💾 Model saved in 'checkpoints/' directory")
        print(f"🎭 Ready for text generation!")
        
    except KeyboardInterrupt:
        print(f"\n" + "="*50)
        print(f"🛑 TRAINING SAFELY STOPPED")
        print(f"💾 All progress has been saved")
        print(f"🔄 To resume training: python train.py --resume")
        print(f"📁 Checkpoints location: checkpoints/")
        print(f"="*50)
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
  