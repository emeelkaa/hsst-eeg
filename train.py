import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from dataset import CHBMITDataset, TUEVDataset
from models import SPaRCNet, TSception, Conformer, BIOTClassifier, HSST
import utils


class Trainer:
    def __init__(self, model, train_dataset, val_dataset, test_dataset,
                 num_classes, batch_size, save_dir, class_counts=None,
                 num_workers=2, lr=3e-4, weight_decay=1e-4):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                                       pin_memory=True, num_workers=num_workers)
        self.val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size,
                                     pin_memory=True, num_workers=num_workers)
        self.test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size,
                                      pin_memory=True, num_workers=num_workers)

        self.model = model.to(self.device)

        if self.num_classes == 1:
            pos_weight = (class_counts[0] / class_counts[1]).to(self.device)
            print(f"[Loss] pos_weight = {pos_weight:.2f}  "
                  f"(non-seizure: {int(class_counts[0])}, seizure: {int(class_counts[1])})")
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        else:
            class_weights = (1.0 / class_counts)
            class_weights = (class_weights / class_weights.sum() * len(class_counts)).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.75, patience=5, verbose=True
        )

    def step(self, batch, training=True):
        inputs, labels = [x.to(self.device) for x in batch]

        if training:
            self.optimizer.zero_grad(set_to_none=True)

        outputs = self.model(inputs)

        if self.num_classes == 1:
            logits = outputs.squeeze(-1)
            loss = self.criterion(logits, labels.float())
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
        else:
            loss = self.criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=-1)
            preds = torch.argmax(probs, dim=-1)

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        return loss.item(), probs, preds, labels

    # ── Full pass over a dataloader ───────────────────────────────────────────
    def evaluate(self, dataloader, training=True):
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss, total_samples = 0.0, 0
        probs_list, preds_list, labels_list = [], [], []

        with torch.set_grad_enabled(training):
            for batch in tqdm(dataloader, desc="Train" if training else "Eval"):
                loss, probs, preds, labels = self.step(batch, training)
                batch_size = labels.size(0)

                total_loss += loss * batch_size
                total_samples += batch_size

                probs_list.append(probs.detach().cpu())
                preds_list.append(preds.detach().cpu())
                labels_list.append(labels.detach().cpu())

        avg_loss = total_loss / total_samples
        all_labels = torch.cat(labels_list).numpy()
        all_probs  = torch.cat(probs_list).numpy()
        all_preds  = torch.cat(preds_list).numpy()

        if self.num_classes == 1:
            acc, balanced_acc, auroc, precision, recall = utils.compute_metrics(
                self.num_classes, all_labels, all_probs, all_preds)
            metrics = {'loss': avg_loss, 'acc': acc, 'bacc': balanced_acc,
                       'auroc': auroc, 'precision': precision, 'recall': recall}
        else:
            acc, balanced_acc, f1, kappa = utils.compute_metrics(
                self.num_classes, all_labels, all_probs, all_preds)
            metrics = {'loss': avg_loss, 'acc': acc, 'bacc': balanced_acc,
                       'f1': f1, 'kappa': kappa}

        return metrics, all_probs, all_preds, all_labels

    def train(self, epochs, patience=15):
        model_path = os.path.join(self.save_dir, 'best_model.pth')
        best_bacc = -np.inf
        patience_counter = 0

        for epoch in range(epochs):
            train_metrics, _, _, _ = self.evaluate(self.train_loader, training=True)

            val_metrics, val_probs, _, val_labels = self.evaluate(self.val_loader, training=False)

            self.scheduler.step(val_metrics['bacc'])

            if self.num_classes == 1:
                print(f"Epoch {epoch+1}/{epochs}"
                      f"\n  Train — Loss: {train_metrics['loss']:.4f}  BAcc: {train_metrics['bacc']:.4f}  AUROC: {train_metrics['auroc']:.4f}"
                      f"\n  Val   — Loss: {val_metrics['loss']:.4f}  BAcc: {val_metrics['bacc']:.4f}  AUROC: {val_metrics['auroc']:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}"
                      f"\n  Train — Loss: {train_metrics['loss']:.4f}  BAcc: {train_metrics['bacc']:.4f}  F1: {train_metrics['f1']:.4f}"
                      f"\n  Val   — Loss: {val_metrics['loss']:.4f}  BAcc: {val_metrics['bacc']:.4f}  F1: {val_metrics['f1']:.4f}")

            if val_metrics['bacc'] > best_bacc:
                best_bacc = val_metrics['bacc']
                torch.save(self.model.state_dict(), model_path)
                print(f"Best model saved with: {val_metrics['bacc']:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

            log_stats = {**{f'train_{k}': float(v) for k, v in train_metrics.items()},
                         **{f'val_{k}': float(v) for k, v in val_metrics.items()},
                         'epoch': epoch}
            with open(os.path.join(self.save_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    def test(self):
        model_path = os.path.join(self.save_dir, 'best_model.pth')
        self.model.load_state_dict(torch.load(model_path))

        test_metrics, _, all_preds, all_labels = self.evaluate(
            self.test_loader, training=False)

        if self.num_classes == 1:
            print(f"\nTest — Loss: {test_metrics['loss']:.4f}  Acc: {test_metrics['acc']:.4f}"
                  f"  BAcc: {test_metrics['bacc']:.4f}"
                  f"\n       AUROC: {test_metrics['auroc']:.4f}  "
                  f"Prec: {test_metrics['precision']:.4f}  Rec: {test_metrics['recall']:.4f}")
        else:
            print(f"\nTest — Loss: {test_metrics['loss']:.4f}  Acc: {test_metrics['acc']:.4f}"
                  f"  BAcc: {test_metrics['bacc']:.4f}"
                  f"\n       F1: {test_metrics['f1']:.4f}  Kappa: {test_metrics['kappa']:.4f}")

        with open(os.path.join(self.save_dir, 'test_metrics.json'), 'w') as f:
            json.dump(test_metrics, f, indent=4)

        if self.num_classes > 1:
            tuev_classes = ['SPSW', 'GPED', 'PLED', 'EYEM', 'ARTF', 'BCKG']
            class_names = tuev_classes if self.num_classes == 6 else None
            self.save_confusion_matrix(all_preds, all_labels, class_names)

        return test_metrics

    # ── Confusion matrix ──────────────────────────────────────────────────────
    def save_confusion_matrix(self, all_preds, all_labels, class_names=None):
        cm = confusion_matrix(all_labels, all_preds)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, data, fmt, title in zip(
            axes,
            [cm, cm_norm],
            ['d', '.2f'],
            ['Confusion Matrix (counts)', 'Confusion Matrix (normalised)']
        ):
            sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues', ax=ax,
                        xticklabels=class_names, yticklabels=class_names)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(title)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'confusion_matrix.png'), dpi=150)
        plt.close(fig)


def get_dataset(dataset='tuev', verbose=True):
    if dataset == 'tuev':
        root = "../../data/tuev/edf"
        train_files = sorted(os.listdir(os.path.join(root, "processed_train")))
        train_sub = sorted(set([f.split("_")[0] for f in train_files]))
        val_sub = np.random.choice(train_sub, size=int(len(train_sub) * 0.1), replace=False)
        train_sub = sorted(set(train_sub) - set(val_sub))

        val_files   = [f for f in train_files if f.split("_")[0] in val_sub]
        train_files = [f for f in train_files if f.split("_")[0] in train_sub]
        test_files  = sorted(os.listdir(os.path.join(root, "processed_eval")))

        train_dataset = TUEVDataset(os.path.join(root, "processed_train"), train_files, 250)
        val_dataset   = TUEVDataset(os.path.join(root, "processed_train"), val_files,   250)
        test_dataset  = TUEVDataset(os.path.join(root, "processed_eval"),  test_files,  250)

        num_channels, num_times, sfreq, num_classes = 16, 1250, 250, 6

        train_labels = [train_dataset[i][1].item() for i in range(len(train_dataset))]
        class_counts = torch.zeros(num_classes)
        for label in train_labels:
            class_counts[int(label)] += 1

    elif dataset == 'chbmit':
        root = "../../data/chbmit/clean_segments"
        train_files = sorted(os.listdir(os.path.join(root, "train")))
        val_files   = sorted(os.listdir(os.path.join(root, "val")))
        test_files  = sorted(os.listdir(os.path.join(root, "test")))

        train_dataset = CHBMITDataset(os.path.join(root, "train"), train_files)
        val_dataset   = CHBMITDataset(os.path.join(root, "val"),   val_files)
        test_dataset  = CHBMITDataset(os.path.join(root, "test"),  test_files)
        num_channels, num_times, sfreq, num_classes = 16, 2000, 200, 1

        class_counts = torch.tensor([49816, 1913], dtype=torch.float)
        print(f"[Dataset] class_counts — 0: {class_counts[0]}  1: {class_counts[1]}")
        print("          Confirm: label 0 = non-seizure, label 1 = seizure")

    if verbose:
        print(f"Train: {len(train_dataset)}  Val: {len(val_dataset)}  Test: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset, num_channels, num_times, sfreq, num_classes, class_counts

def get_model(model_name, num_channels, num_times, num_classes, sfreq,
              emb_size, depth, num_heads):
    if model_name == 'sparcnet':
        return SPaRCNet(num_channels=num_channels, num_timepoints=num_times,
                        num_classes=num_classes)
    elif model_name == 'tsception':
        return TSception(num_channels=num_channels, sfreq=sfreq,
                         num_classes=num_classes)
    elif model_name == 'conformer':
        return Conformer(emb_size=emb_size, depth=depth, num_heads=num_heads,
                         num_classes=num_classes, num_channels=num_channels)
    elif model_name == 'biot':
        return BIOTClassifier(emb_size=emb_size, depth=depth, num_heads=num_heads,
                              num_classes=num_classes, num_channels=num_channels,
                              n_fft=sfreq, hop_length=sfreq // 2)
    elif model_name == 'hsst':
        return HSST(emb_size=emb_size, depth=depth, num_heads=num_heads,
                    num_classes=num_classes, num_channels=num_channels)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    dataset_name = 'tuev'
    model_name   = 'hsst'
    emb_size     = 64
    depth        = 6
    num_heads    = 4
    batch_size   = 32
    num_workers  = 2

    if dataset_name == 'tuev':
        seeds = [1, 42, 100, 1000, 12345]
        start_time = time.time()
        for seed in seeds:
            set_seed(seed)
            (train_dataset, val_dataset, test_dataset,
             num_channels, num_times, sfreq,
             num_classes, class_counts) = get_dataset(dataset_name, verbose=True)

            model = get_model(model_name, num_channels, num_times, num_classes,
                              sfreq, emb_size, depth, num_heads)

            trainer = Trainer(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                batch_size=batch_size,
                num_classes=num_classes,
                num_workers=num_workers,
                save_dir=f"outputs/{dataset_name}/{model_name}/{seed}/",
                class_counts=class_counts,
            )
            trainer.train(epochs=50, patience=10)
            test_metrics = trainer.test()
            print(f"Seed {seed}: {test_metrics}")

        print(f"Done in {time.time() - start_time:.1f}s")

    elif dataset_name == 'chbmit':
        seeds = [42, 100, 12345]
        models = ['sparcnet', 'tsception', 'conformer', 'biot', 'hsst']
        start_time = time.time()
        for model_name in models:
            for seed in seeds:
                set_seed(seed)
                (train_dataset, val_dataset, test_dataset,
                num_channels, num_times, sfreq,
                num_classes, class_counts) = get_dataset(dataset_name, verbose=True)

                model = get_model(model_name, num_channels, num_times, num_classes,
                                    sfreq, emb_size, depth, num_heads)

                trainer = Trainer(
                    model=model,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    test_dataset=test_dataset,
                    batch_size=batch_size,
                    num_classes=num_classes,
                    num_workers=num_workers,
                    save_dir=f"outputs/{dataset_name}/{model_name}/{seed}/",
                    class_counts=class_counts,
                )
                trainer.train(epochs=50, patience=10) 
                test_metrics = trainer.test()
                print(f"Test: {test_metrics}")