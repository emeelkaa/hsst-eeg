import os
import json
import torch
import logging
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pyhealth.metrics import binary_metrics_fn
from pyhealth.metrics import multiclass_metrics_fn
from tqdm import tqdm

from dataset import get_chbmit, get_tuev
from models import EEGConformer, BIOT, HSST, SPARCNet, TSception

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Trainer: 
    def __init__(self, model: torch.nn.Module, train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset, num_classes: int, 
                 batch_size: int, num_workers: int, lr: float = 1e-3, weight_decay: float = 1e-4, save_dir: str = "results", seed: int = 42
    ):
        # Set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=num_workers)
        self.val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=num_workers)
        self.test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=num_workers)

        self.model = model.to(self.device)

        if self.num_classes == 1:
            self.criterion = torch.nn.BCEWithLogitsLoss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3)

        if self.num_classes == 1:
            self.history = {
                'train_loss': [], 'train_acc': [], 'train_bacc': [], 'train_pr_auc': [], 'train_auroc': [],
                'val_loss': [], 'val_acc': [], 'val_bacc': [], 'val_pr_auc': [], 'val_auroc': []
            }
        else:
            self.history = {
                'train_loss': [], 'train_bacc': [], 'train_f1': [], 'train_kappa': [],
                'val_loss': [], 'val_bacc': [], 'val_f1': [], 'val_kappa': []
            }

    def compute_metrics(self, labels, probs):
        if self.num_classes == 1:
            metrics = binary_metrics_fn(labels, probs, metrics=['accuracy', 'balanced_accuracy', 'pr_auc', 'roc_auc'])
            return metrics['accuracy'], metrics['balanced_accuracy'], metrics['pr_auc'], metrics['roc_auc']
        else:
            metrics = multiclass_metrics_fn(labels, probs, metrics=['balanced_accuracy', 'f1_weighted', 'cohen_kappa'])
            return metrics['balanced_accuracy'], metrics['f1_weighted'], metrics['cohen_kappa']

    def save_history(self):
        with open(os.path.join(self.save_dir, 'history.json'), 'w') as f:
            json.dump(self.history, f, indent=4)

    def step(self, batch, training=True):
        inputs, labels = [x.to(self.device) for x in batch]
        
        if training:
            self.optimizer.zero_grad(set_to_none=True)

        outputs = self.model(inputs)

        if self.num_classes == 1:
            logits = outputs.squeeze(-1)
            loss = self.criterion(logits, labels.float())
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
        else:
            loss = self.criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=-1)
            preds = torch.argmax(probs, dim=-1)

        if training: 
            loss.backward()
            self.optimizer.step()
        
        correct = (preds == labels).sum().item()
        return loss.item(), correct, probs, labels
    

    def evaluate(self, dataloader, training=True):
        if training: 
            self.model.train()
        else:
            self.model.eval()

        total_loss, total_correct, total_samples  = 0.0, 0, 0
        probs_list, labels_list = [], []

        with torch.set_grad_enabled(training):
            for batch in tqdm(dataloader, desc="Training" if training else "Evaluating"):
                loss, correct, probs, labels = self.step(batch, training)
                batch_size = labels.size(0)

                total_loss += loss * batch_size
                total_correct += correct
                total_samples += batch_size

                probs_list.append(probs.detach().cpu())
                labels_list.append(labels.detach().cpu())
        
        avg_loss = total_loss / total_samples

        all_probs = torch.cat(probs_list).numpy()
        all_labels = torch.cat(labels_list).numpy()

        if self.num_classes == 1:
            acc, balanced_acc, pr_auc, auroc = self.compute_metrics(all_labels, all_probs)
            return avg_loss, acc, balanced_acc, pr_auc, auroc
        else:
            balanced_acc, f1, kappa = self.compute_metrics(all_labels, all_probs)
            return avg_loss, balanced_acc, f1, kappa
            

    def train(self, epochs, patience):
        model_path = os.path.join(self.save_dir, 'best_model.pth') 
        best_loss, patience_counter = float('inf'), 0

        for epoch in range(epochs):
            if self.num_classes == 1:
                train_loss, train_acc, train_bacc, train_pr_auc, train_auroc = self.evaluate(self.train_loader, training=True)
                val_loss, val_acc, val_bacc, val_pr_auc, val_auroc = self.evaluate(self.val_loader, training=False)

                self.scheduler.step(val_loss)

                for key, val in zip(['train_loss', 'train_acc', 'train_bacc', 'train_pr_auc', 'train_auroc',
                                    'val_loss', 'val_acc', 'val_bacc', 'val_pr_auc', 'val_auroc'],
                                    [train_loss, train_acc, train_bacc, train_pr_auc, train_auroc,
                                    val_loss, val_acc, val_bacc, val_pr_auc, val_auroc]):
                    self.history[key].append(val)
                
                logging.info(f"Epoch {epoch+1}/{epochs}"
                            f"\nTrain - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, BAcc: {train_bacc:.2f}%"
                            f"\nAUROC: {train_auroc:.4f}, PR-AUC: {train_pr_auc:.4f}"
                            f"\nVal - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, BAcc: {val_bacc:.2f}%"
                            f"\nAUROC: {val_auroc:.4f}, PR-AUC: {val_pr_auc:.4f}")

                self.save_history()
            
            else:
                train_loss, train_bacc, train_f1, train_kappa = self.evaluate(self.train_loader, training=True)
                val_loss, val_bacc, val_f1, val_kappa = self.evaluate(self.val_loader, training=False)

                self.scheduler.step(val_loss)

                for key, val in zip(['train_loss', 'train_bacc', 'train_f1', 'train_kappa',
                                    'val_loss', 'val_bacc', 'val_f1', 'val_kappa'],
                                    [train_loss, train_bacc, train_f1, train_kappa,
                                    val_loss, val_bacc, val_f1, val_kappa]):
                    self.history[key].append(val)
                
                logging.info(f"Epoch {epoch+1}/{epochs}"
                            f"\nTrain - Loss: {train_loss:.4f}, BACC: {train_bacc:.2f}%, F1: {train_f1:.2f}%, Kappa: {train_kappa:.2f}" 
                            f"\nVal - Loss: {val_loss:.4f}, BACC: {val_bacc:.2f}%, F1: {val_f1:.2f}%, Kappa: {val_kappa:.2f}"
                )

                self.save_history()

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), model_path)
                logging.info(f"Best model saved with val loss: {best_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"Early stopping at epoch {epoch+1}")
                    break

    
    def test(self):
        model_path = os.path.join(self.save_dir, 'best_model.pth')
        self.model.load_state_dict(torch.load(model_path))

        if self.num_classes == 1:
            test_loss, test_acc, test_bacc, test_pr_auc, test_auroc = self.evaluate(self.test_loader, training=False)
            logging.info(f"Test - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%, BAcc: {test_bacc:.2f}%"
                    f"\nAUROC: {test_auroc:.4f}, PR-AUC: {test_pr_auc:.4f}")
            
            test_metrics = {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_bacc': test_bacc,
            'test_pr_auc': test_pr_auc,
            'test_auroc': test_auroc,
            }
        else:
            test_loss, test_bacc, test_f1, test_kappa = self.evaluate(self.test_loader, training=False)
            logging.info(f"Test - Loss: {test_loss:.4f}, BACC: {test_bacc:.2f}%, F1: {test_f1:.2f}%, Kappa: {test_kappa:.2f}"
            )
            
            test_metrics = {
            'test_loss': test_loss,
            'test_bacc': test_bacc,
            'test_f1': test_f1,
            'test_kappa': test_kappa,            
            }

        with open(os.path.join(self.save_dir, 'test_metrics.json'), 'w') as f:
            json.dump(test_metrics, f, indent=4)



if __name__ == "__main__":
    datasets = ['chbmit_2']
    
    for dataset in datasets:
        if dataset == 'chbmit_2':
            train_dataset, val_dataset, test_dataset = get_chbmit()

            n_channels = 16
            n_times = 2560
            sfreq = 256
            n_classes = 1
        elif dataset == 'tuev':
            train_dataset, val_dataset, test_dataset = get_tuev()

            n_channels = 16
            n_times = 1250 
            sfreq = 250
            n_classes = 6
      
        models = ['biot', 'conformer', 'hsst', 'sparcnet', 'tsception']
        emb_size = 64
        num_heads = 4
        depth = 2

        for i, model_name in enumerate(models):
            if model_name == 'conformer':
                model = EEGConformer(emb_size=emb_size, depth=depth, num_heads=num_heads, n_channels=n_channels, n_classes=n_classes)
            elif model_name == 'biot':
                model = BIOT(emb_size=emb_size, depth=depth, num_heads=num_heads, n_channels=n_channels, n_classes=n_classes)
            elif model_name == 'hsst':
                model = HSST(emb_size=emb_size, depth=depth//2, num_heads=num_heads, n_channels=n_channels, n_classes=n_classes)
            elif model_name == 'sparcnet':
                model = SPARCNet(num_channels=n_channels, num_times=n_times, num_classes=n_classes)
            elif model_name == 'tsception':
                model = TSception(num_classes=n_classes, input_size=(1, n_channels, n_times), sampling_rate=sfreq, num_T=32, num_S=32, hidden=64, dropout_rate=0.3)
            
            seeds = [1234]

            for seed in seeds:
                trainer = Trainer(
                    model=model,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    test_dataset=test_dataset,
                    batch_size=16,
                    num_classes=n_classes,
                    num_workers=2,
                    save_dir=f"results/{dataset}/{model_name}/{seed}",
                    seed=seed 
                )

                trainer.train(epochs=100, patience=10)
                trainer.test()