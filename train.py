import os
import torch
import wandb
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import get_scheduler
from Bio import SeqIO
from typing import List, Tuple, Optional
import numpy as np
from torch import nn
from carmania.tokenization_carmania import CarmaniaTokenizer
from carmania.modeling_carmania import CarmaniaModel
from carmania.configuration_carmania import CarmaniaConfig
from carmania.loss import TMLoss

class DNASequenceDataset(Dataset):
    def __init__(self, fasta_path: str, tokenizer: CarmaniaTokenizer, max_sequences: Optional[int] = None):
        self.input_ids = []
        self.bigrams = []

        print("Loading and tokenizing sequences...")
        for i, record in enumerate(SeqIO.parse(fasta_path, "fasta")):
            if max_sequences and i >= max_sequences:
                break
            seq = str(record.seq)
            token_ids, bigram_matrix = tokenizer.encode_with_bigram(seq)
            self.input_ids.append(token_ids)
            self.bigrams.append(bigram_matrix)

        self.input_ids = torch.tensor(self.input_ids, dtype=torch.long)
        self.bigrams = torch.tensor(self.bigrams, dtype=torch.float32)
        print(f"Loaded {len(self.input_ids)} sequences.")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "bigrams": self.bigrams[idx]
        }

# === Training function ===
def train(
    fasta_path,
    max_sequences=100_000,
    batch_size=16,
    epochs=4,
    learning_rate=5e-4,
    model_name="carmania",
    beta= 1 ,
    device="cuda:0"
):
    wandb.init(project="carmania", name=model_name)

    # Setup
    config = CarmaniaConfig()
    tokenizer = CarmaniaTokenizer(model_max_length=config.seq_length, calculate_bigram=True)
    dataset = DNASequenceDataset(fasta_path, tokenizer, max_sequences=max_sequences)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model = CarmaniaModel(config).to(device)
    TM_loss = TMLoss()
    NT_loss = nn.CrossEntropyLoss(ignore_index=4) # [PAd]==4

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler()
    scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=100, num_training_steps=len(dataloader)*epochs)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            bigrams = batch["bigrams"].to(device)

            with autocast():
                logits = model(input_ids[:, :-1])
                
                loss1 = NT_loss( 
                    logits.reshape(-1, logits.size(-1)),
                    input_ids[:, 1:].reshape(-1))
                
                loss2 = TM_loss(logits, bigrams)

                loss = loss1 + loss2*beta  

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            total_loss += loss.item()
            if step % 20 == 0:
                wandb.log({"step_loss": loss.item(), "epoch": epoch+1 , "tm_loss": loss2.item() , "nt_loss":loss1.item(),"learning_rate": scheduler.get_last_lr()[0]})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")
        wandb.log({"epoch": epoch+1, "avg_loss": avg_loss})
        torch.save(model.state_dict(), f"{model_name}_epoch{epoch+1}.pt")

    wandb.finish()

# === Run ===
if __name__ == "__main__":
    fasta_path = "./human_sequences.fa" 
    train(fasta_path)
