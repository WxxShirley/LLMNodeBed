"""Generate embeddings for ogbn-products with batched processing to manage memory"""
import torch
import numpy as np
from tqdm import tqdm
import os
import argparse
import sys
import gc
sys.path.append("../..")
from common import get_cur_time, TextEncoder


def generate_embeddings_in_batches(texts, text_encoder, batch_size=10000, save_dir="../../datasets/tmp_embeddings", 
                                  prefix="ogbn-products", pooling="mean"):
    """
    Generate embeddings for texts in batches and save intermediate results
    
    Args:
        texts: List of text strings
        text_encoder: TextEncoder instance
        batch_size: Number of texts to process in one batch
        save_dir: Directory to save intermediate results
        prefix: Prefix for saved files
        pooling: Pooling strategy ('cls' or 'mean')
    
    Returns:
        List of saved file paths
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Find the last processed batch if any
    existing_files = [f for f in os.listdir(save_dir) if f.startswith(f"{prefix}_") and f.endswith(".npy")]
    existing_batches = [int(f.split('_')[1].split('.')[0]) for f in existing_files if f.split('_')[1].split('.')[0].isdigit()]
    
    start_batch = max(existing_batches) + 1 if existing_batches else 0
    start_idx = start_batch * batch_size
    
    if start_idx >= len(texts):
        print(f"All batches already processed. Starting from the beginning.")
        start_batch = 0
        start_idx = 0
    
    saved_files = [f"{save_dir}/{prefix}_{i}.npy" for i in range(start_batch)]
    
    # Process texts in batches
    with torch.no_grad():
        for batch_idx in range(start_batch, (len(texts) + batch_size - 1) // batch_size):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(texts))
            
            print(f"Processing batch {batch_idx+1}/{(len(texts) + batch_size - 1) // batch_size} " 
                  f"(texts {batch_start} to {batch_end-1})")
            
            text_embeddings = []
            for text in tqdm(texts[batch_start:batch_end], desc=f"Batch {batch_idx}"):
                text_emb = text_encoder(input_text=text, pooling=pooling)
                # Convert to numpy to save memory
                text_embeddings.append(text_emb.cpu().numpy())
                
                # Clear GPU memory after each text
                torch.cuda.empty_cache()
            
            # Convert list of embeddings to single numpy array
            batch_embeddings = np.vstack(text_embeddings)
            
            # Save batch results
            save_path = f"{save_dir}/{prefix}_{batch_idx}.npy"
            np.save(save_path, batch_embeddings)
            saved_files.append(save_path)
            
            # Force garbage collection
            del text_embeddings, batch_embeddings
            gc.collect()
            torch.cuda.empty_cache()
            
            print(f"Saved batch {batch_idx} to {save_path}")
    
    return saved_files


def combine_embeddings(file_paths, output_path, device="cuda:0"):
    """
    Combine multiple numpy embedding files into a single tensor file
    
    Args:
        file_paths: List of paths to numpy embedding files
        output_path: Path to save the combined tensor
        device: Device to use
    """
    print(f"Combining {len(file_paths)} embedding files...")
    
    # Load and combine embeddings
    all_embeddings = []
    for file_path in tqdm(file_paths, desc="Loading files"):
        embeddings = np.load(file_path)
        all_embeddings.append(embeddings)
    
    # Stack all embeddings
    combined_embeddings = np.vstack(all_embeddings)
    
    # Convert to tensor and save
    tensor_embeddings = torch.tensor(combined_embeddings, device="cpu")
    torch.save(tensor_embeddings, output_path)
    
    print(f"Combined embeddings saved to {output_path}")
    print(f"Embedding shape: {tensor_embeddings.shape}")
    
    return tensor_embeddings.shape


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--encoder_name", type=str, default="Qwen-3B", 
                        choices=["MiniLM", "SentenceBert", "e5-large", "roberta", 
                                "Qwen-3B", "Qwen-7B", "Mistral-7B", "Llama-8B"])
    parser.add_argument("--use_cls", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--save_emb", type=int, default=1)
    parser.add_argument("--only_combine", type=int, default=1, 
                        help="Skip generation and only combine existing files")
    
    args = parser.parse_args()
    
    print('= ' * 20)
    print('## Starting Time:', get_cur_time(), flush=True)
    print(args, "\n")
    
    device = torch.device(args.device)
    dataset_name = "ogbn-products"
    
    # Directory to save intermediate results
    temp_dir = f"../../datasets/tmp_embeddings"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Final output path
    output_dir = f"../../datasets/{args.encoder_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{dataset_name}.pt"
    
    if os.path.exists(output_path):
        print(f"[{dataset_name}-{args.encoder_name}] Embedding file already exists, Quit!")
        print('= ' * 20)
        exit()
    
    # If only combining existing files, skip generation
    if args.only_combine == 0:
        # Load the graph data
        print(f"Loading {dataset_name} dataset...")
        graph_data = torch.load(f"../../datasets/{dataset_name}.pt")
        
        # Initialize encoder
        encoder_type = "LM" if args.encoder_name in ["MiniLM", "SentenceBert", "e5-large", "roberta"] else "LLM"
        text_encoder = TextEncoder(encoder_name=args.encoder_name, encoder_type=encoder_type, device=device)
        
        # Generate embeddings in batches
        saved_files = generate_embeddings_in_batches(
            texts=graph_data.raw_texts,
            text_encoder=text_encoder,
            batch_size=args.batch_size,
            save_dir=temp_dir,
            prefix=dataset_name,
            pooling="mean",
        )
    else:
        # Get all existing batch files
        saved_files = sorted([
            f"{temp_dir}/{f}" for f in os.listdir(temp_dir) 
            if f.startswith(f"{dataset_name}_") and f.endswith(".npy")
        ], key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        print(f"Found {len(saved_files)} existing batch files")
    
    # Combine all batches into a single tensor file if needed
    if args.save_emb:
        emb_shape = combine_embeddings(saved_files, output_path, device=device)
        print(f"Final embedding shape: {emb_shape}")
    
    print('\n## Finishing Time:', get_cur_time(), flush=True)
    print('= ' * 20)
    print("Done!")
