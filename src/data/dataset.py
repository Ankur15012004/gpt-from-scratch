import torch 
from torch.utils.data import Dataset

class TextDataset(Dataset):
    """
    Custom dataset for loading text data and preparing training batches
    
    Think of this as a "text processor" that:
    1. Takes raw text
    2. Converts it to numbers (tokens)
    3. Creates training pairs for next-token prediction
    """

    def __init__(self,text: str,tokenizer,block_size: int):
        """
        Initialize our dataset
        
        Args:
            text: Raw text string (like "Hello world! This is amazing.")
            tokenizer: Our tokenizer from previous step
            block_size: How many tokens in each training sequence
        
        Example:
            If text = "Hello world! This is amazing." and block_size = 4
            We might get sequences like:
            - [Hello, world, !, This] → [world, !, This, is]
            - [world, !, This, is] → [!, This, is, amazing]
        """
        self.tokenizer=tokenizer
        self.block_size=block_size
        # Convert entire text to token IDs using our tokenizer
        print(f"📝 Tokenizing text...")
        self.token_ids = self.tokenizer.encode(text)
        print(f"✅ Text converted to {len(self.token_ids)} tokens")

        # Calculate how many complete sequences we can create
        # We need block_size + 1 tokens to create one training pair
        # (input sequence + next token as target)
        self.num_sequences = len(self.token_ids) - self.block_size

        if self.num_sequences<=0:
            raise ValueError(f"Text is too short! Need at least {self.block_size+1} tokens, got {len(self.token_ids)}")
        
        print(f"📊 Created {self.num_sequences} training sequences")

    def __len__(self):
        """
        How many training sequences do we have?
        
        PyTorch needs this method to know how big our dataset is.
        Think of it as "How many pages are in our training book?"
        """
        return self.num_sequences
    
    def __getitem__(self, idx):
        """
        Get one training example by index
        
        This is where the magic happens! We create input-target pairs.
        
        Args:
            idx: Which training sequence to get (0, 1, 2, ...)
            
        Returns:
            Dictionary with:
            - input_ids: sequence of tokens to feed to model
            - target_ids: what the model should predict (shifted by 1)
            
        Example:
            If our tokens are [10, 20, 30, 40, 50] and block_size=3, idx=0:
            - input_ids:  [10, 20, 30]  # "Give me the next token after these"
            - target_ids: [20, 30, 40]  # "These are the correct next tokens"
        """

        start_idx=idx

        input_sequence=self.token_ids[start_idx:start_idx+self.block_size]

        target_sequence=self.token_ids[start_idx+1:start_idx+self.block_size+1]

        # Convert to PyTorch tensors (fancy arrays that GPUs can process)
        input_ids = torch.tensor(input_sequence, dtype=torch.long)
        target_ids = torch.tensor(target_sequence, dtype=torch.long)

        return {
            'input_ids': input_ids,    # What we give to the model
            'target_ids': target_ids   # What the model should predict
        }
    
    @classmethod
    def from_file(cls,file_path:str,tokenizer,block_size:int):
        """Create dataset from text file"""

        try:
            with open(file_path,"r",encoding='utf-8') as f:
                text=f.read()

            print(f"📖 Loaded text file: {file_path}")
            print(f"📏 File length: {len(text)} characters")

            return cls(text,tokenizer,block_size)

            

        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find file: {file_path}")
        except Exception as e:
            raise Exception(f"Error reading file {file_path}: {str(e)}")
        
def test_dataset():
    """Test our dataset implementation"""
    print("🧪 Testing TextDataset...")

    from src.data.tokenizer import GPTTokenizer

    tokenizer=GPTTokenizer()
    test_text= "Hello world! This is a test dataset for transformer training."
    print(f"📝 Test text: '{test_text}'")

    block_size=4
    dataset= TextDataset(test_text,tokenizer,block_size)

    print(f"\n📊 Dataset Info:")
    print(f"   - Block size: {block_size}")
    print(f"   - Number of sequences: {len(dataset)}")

    print(f"\n🔍 First few training examples:")

    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        
        input_text = tokenizer.decode(sample['input_ids'].tolist()) ## because decode method requires list of tokens ids they are tensor right now 
        target_text = tokenizer.decode(sample['target_ids'].tolist())
        
        print(f"\n   Example {i}:")
        print(f"   Input text: '{input_text}'")
        print(f"   Target text: '{target_text}'")
        print(f"   → Learn: given '{input_text}', predict '{target_text}'")

if __name__ == "__main__":
    test_dataset()


        


