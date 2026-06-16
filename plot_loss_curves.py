import json
import matplotlib.pyplot as plt

def plot_loss_curves(history_file):
    print(f"Loading history from {history_file}")
    with open(history_file, 'r') as f:
        history = json.load(f)

    epochs = [item['epoch'] for item in history]
    train_loss = [item['train_loss'] for item in history]
    val_mse = [item['val_mse'] for item in history]

    print("Plotting loss curves...")
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_mse, label='Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    
    save_path = 'loss_curves.png'
    plt.savefig(save_path)
    print(f"Loss curves saved to {save_path}")

if __name__ == '__main__':
    plot_loss_curves('C:/Users/ylchen/workspace/research_group/artifacts/finetune_checkpoints/exp_5/history.json')
