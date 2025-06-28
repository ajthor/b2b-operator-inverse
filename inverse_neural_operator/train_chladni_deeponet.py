import torch
import numpy as np
import sys 
import os 
from datetime import datetime 
import argparse
from tqdm import trange

# Add the project root directory to sys.path
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import required modules
from deeponet_model import DeepONet, MLP
from data.chladni_2d import load_data
import torch.nn as nn

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train DeepONet for Chladni plate problem.")
    
    parser.add_argument('--data_path', type=str, default="Data/chladni_dataset", help='Path to dataset')
    parser.add_argument('--don_p_dim', type=int, default=64, help='Latent dimension p')
    parser.add_argument('--don_branch_hidden', type=int, default=256, help='Branch network hidden size')
    parser.add_argument('--don_trunk_hidden', type=int, default=256, help='Trunk network hidden size')
    parser.add_argument('--don_n_branch_layers', type=int, default=3, help='Branch network layers')
    parser.add_argument('--don_n_trunk_layers', type=int, default=4, help='Trunk network layers')
    parser.add_argument('--activation_fn', type=str, default="relu", choices=["relu", "tanh", "gelu", "swish"], help='Activation function')
    parser.add_argument('--don_lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--don_epochs', type=int, default=50000, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument("--lr_schedule_steps", type=int, nargs='+', default=[25000, 75000, 125000, 175000], help="LR decay steps")
    parser.add_argument("--lr_schedule_gammas", type=float, nargs='+', default=[0.2, 0.5, 0.2, 0.5], help="LR decay factors")
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to pre-trained model')
    
    return parser.parse_args()

def setup_logging(project_root):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(project_root, "logs", "DeepONet_chladni", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to: {log_dir}")
    return log_dir

def get_activation_function(activation_name):
    activation_map = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'gelu': nn.GELU, 'swish': nn.SiLU}
    return activation_map.get(activation_name.lower(), nn.ReLU)

class ChladniDataLoader:
    """Data loader for Chladni dataset with fast tensor indexing."""
    
    def __init__(self, dataset, batch_size=64):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_samples = len(dataset)
        self.device = dataset.device
        
        # Extract tensors for fast indexing (like the old implementation)
        self.u_data = dataset.u.squeeze(-1) if dataset.u.dim() == 3 else dataset.u  # [N, 625]
        self.Y_data = dataset.Y  # [N, 625, 2]
        self.s_data = dataset.s.squeeze(-1) if dataset.s.dim() == 3 else dataset.s  # [N, 625]
        
    def sample(self):
        """Sample a batch from the dataset using fast tensor indexing."""
        indices = torch.randint(0, self.n_samples, (self.batch_size,), device=self.device)
        # Direct tensor indexing - much faster than looping through __getitem__
        return self.u_data[indices], self.Y_data[indices], self.s_data[indices]

def create_model(args, device):
    activation_fn = get_activation_function(args.activation_fn)
    
    branch_net = MLP(625, [args.don_branch_hidden] * args.don_n_branch_layers, args.don_p_dim, activation_fn)
    trunk_net = MLP(2, [args.don_trunk_hidden] * args.don_n_trunk_layers, args.don_p_dim, activation_fn)
    
    return DeepONet(branch_net, trunk_net).to(device)

def train_model(model, train_loader, args, device, log_dir):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.don_lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_schedule_steps, gamma=0.5)
    criterion = nn.MSELoss()
    
    model.train()
    epoch_losses = []
    
    bar = trange(args.don_epochs)
    for epoch in bar:
        current_lr = optimizer.param_groups[0]['lr']
        branch_input, trunk_input, target = train_loader.sample()
        
        optimizer.zero_grad()
        pred = model(branch_input, trunk_input)
        loss = criterion(pred, target)
        
        with torch.no_grad():
            rel_l2_error = torch.norm(pred - target) / torch.norm(target)
        
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        epoch_losses.append(loss.item())
        bar.set_description(f"Step {epoch + 1} | Loss: {loss.item():.4e} | Rel L2: {rel_l2_error.item():.4f} | Grad Norm: {grad_norm:.2f} | LR: {current_lr:.2e}")
        
        if epoch % 25000 == 0 and epoch > 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'loss': loss.item()}, 
                      os.path.join(log_dir, f"deeponet_checkpoint_epoch_{epoch}.pth"))
    
    return epoch_losses

def evaluate_model(model, test_dataset, device, n_test_samples=100):
    model.eval()
    n_test = min(n_test_samples, len(test_dataset))
    
    # Use fast tensor indexing for evaluation too
    u_data = test_dataset.u.squeeze(-1) if test_dataset.u.dim() == 3 else test_dataset.u
    Y_data = test_dataset.Y
    s_data = test_dataset.s.squeeze(-1) if test_dataset.s.dim() == 3 else test_dataset.s
    
    # Process in batches for memory efficiency
    batch_size = 32
    total_loss = total_rel_error = 0.0
    
    with torch.no_grad():
        for i in range(0, n_test, batch_size):
            end_idx = min(i + batch_size, n_test)
            batch_indices = torch.arange(i, end_idx, device=device)
            
            # Fast tensor indexing
            u_batch = u_data[batch_indices]
            Y_batch = Y_data[batch_indices] 
            s_batch = s_data[batch_indices]
            
            pred_batch = model(u_batch, Y_batch)
            
            # Compute losses for the batch
            batch_loss = torch.nn.MSELoss()(pred_batch, s_batch).item()
            batch_rel_error = (torch.norm(pred_batch - s_batch) / torch.norm(s_batch)).item()
            
            total_loss += batch_loss * (end_idx - i)
            total_rel_error += batch_rel_error * (end_idx - i)
    
    avg_loss, avg_rel_error = total_loss / n_test, total_rel_error / n_test
    print(f"Test Results - MSE Loss: {avg_loss:.6e}, Relative Error: {avg_rel_error:.6f}")
    
    model.train()
    return avg_loss, avg_rel_error

def main():
    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_dir = setup_logging(project_root)
    
    try:
        # Load training and test datasets using the new format
        print("Loading training dataset...")
        train_dataset = load_data(device=str(device), split="train")
        print("Loading test dataset...")
        test_dataset = load_data(device=str(device), split="test")
        
        print(f"Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Create data loader
    train_loader = ChladniDataLoader(train_dataset, batch_size=args.batch_size)
    
    model = create_model(args, device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    if args.load_model_path:
        checkpoint = torch.load(args.load_model_path, map_location=device)
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        print("Pre-trained model loaded")
    
    train_model(model, train_loader, args, device, log_dir)
    evaluate_model(model, test_dataset, device, n_test_samples=1000)
    
    torch.save(model.state_dict(), os.path.join(log_dir, "deeponet_model.pth"))
    print("Training completed!")

if __name__ == "__main__":
    main() 