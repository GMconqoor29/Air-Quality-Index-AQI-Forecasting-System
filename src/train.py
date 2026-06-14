import torch
import torch.nn as nn
from src.model import LSTMAQIModel
import src.config as config

def evaluate_model(model, loader, criterion, device):
    model.eval()
    losses, all_out, all_y = [], [], []
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device).view(-1, 1)
            out = model(X)
            losses.append(criterion(out, y).item() * X.size(0))
            all_out.append(out)
            all_y.append(y)
            
    avg_loss = sum(losses) / len(loader.dataset)
    return avg_loss, torch.cat(all_out).cpu().numpy(), torch.cat(all_y).cpu().numpy()

def train_model(model, train_loader, val_loader, criterion, optimizer, device,
                epochs=config.EPOCHS, patience=5, checkpoint_path='best_model.pt',
                verbose=True, trial=None):
    best_val = float("inf")
    counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        # Phase Train
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device).view(-1, 1)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)
        
        t_loss = train_loss / len(train_loader.dataset)
        v_loss, _, _ = evaluate_model(model, val_loader, criterion, device)
        
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        
        # Report validation loss to Optuna to support trial pruning
        if trial is not None:
            import optuna
            trial.report(v_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        if verbose:
            print(f"Epoch {epoch:02d}: Train MSE {t_loss:.4f} | Val MSE {v_loss:.4f}")

        # Check Early Stopping
        if v_loss < best_val:
            best_val = v_loss
            counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            counter += 1
            if counter >= patience:
                if verbose:
                    print("==> Early stopping triggered")
                break

    if verbose:
        print("Loading best model weights...")
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    return model, history
