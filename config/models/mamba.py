import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from mamba_ssm import Mamba
import warnings

warnings.filterwarnings("ignore")

class SmartContractDataset(Dataset):
    """Dataset PyTorch pour les fragments de smart contracts"""
    def __init__(self, vectors, labels):
        self.vectors = torch.FloatTensor(vectors)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.vectors)
    
    def __getitem__(self, idx):
        return self.vectors[idx], self.labels[idx]

class MambaClassifier(nn.Module):
    """Modèle de classification basé sur Mamba"""
    def __init__(self, args):
        super(MambaClassifier, self).__init__()
        
        self.args = args
        self.d_model = args.d_model
        self.n_layers = args.n_layers
        self.num_classes = args.num_classes
        
        # Projection d'entrée pour adapter la dimension des vecteurs Word2Vec
        self.input_projection = nn.Linear(args.vec_length, self.d_model)
        
        # Couches Mamba empilées
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=self.d_model,
                d_state=args.d_state,
                d_conv=args.d_conv,
                expand=args.expand,
            ) for _ in range(self.n_layers)
        ])
        
        # Normalisation par couche
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.d_model) for _ in range(self.n_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(args.dropout)
        
        # Tête de classification
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(self.d_model // 2, self.d_model // 4),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(self.d_model // 4, self.num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, vec_length)
        batch_size, seq_len, _ = x.shape
        
        # Projection d'entrée
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Passage à travers les couches Mamba
        for i, (mamba_layer, layer_norm) in enumerate(zip(self.mamba_layers, self.layer_norms)):
            # Connexion résiduelle avec normalisation
            residual = x
            x = layer_norm(x)
            x = mamba_layer(x) + residual
            x = self.dropout(x)
        
        # Agrégation globale (moyenne sur la dimension de séquence)
        x = torch.mean(x, dim=1)  # (batch_size, d_model)
        
        # Classification
        logits = self.classifier(x)  # (batch_size, num_classes)
        
        return logits

class MambaTrainer:
    """Classe pour l'entraînement du modèle Mamba"""
    def __init__(self, data, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Utilisation du device: {self.device}")
        
        # Préparation des données
        self.prepare_data(data)
        
        # Initialisation du modèle
        self.model = MambaClassifier(args).to(self.device)
        print(f"Modèle créé avec {sum(p.numel() for p in self.model.parameters())} paramètres")
        
        # Optimiseur et scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs
        )
        
        # Fonction de perte avec poids de classe
        self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(self.class_weights).to(self.device))
    
    def prepare_data(self, data):
        """Préparation des données d'entraînement et de test"""
        vectors = np.stack(data.iloc[:, 0].values)
        labels = data.iloc[:, 1].values
        
        # Filtrage des échantillons positifs et négatifs
        positive_idxs = np.where(labels == 1)[0]
        negative_idxs = np.where(labels == 0)[0]
        idxs = np.concatenate([positive_idxs, negative_idxs])
        
        # Division train/test
        x_train, x_test, y_train, y_test = train_test_split(
            vectors[idxs], labels[idxs],
            test_size=0.2, stratify=labels[idxs], random_state=42
        )
        
        # Calcul des poids de classe
        classes = np.unique(y_train)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_train
        )
        self.class_weights = class_weights
        
        # Création des datasets
        self.train_dataset = SmartContractDataset(x_train, y_train)
        self.test_dataset = SmartContractDataset(x_test, y_test)
        
        # Création des dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        print(f"Données préparées:")
        print(f"  - Entraînement: {len(self.train_dataset)} échantillons")
        print(f"  - Test: {len(self.test_dataset)} échantillons")
        print(f"  - Poids de classe: {self.class_weights}")
    
    def train_epoch(self):
        """Entraînement pour une époque"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {100. * correct / total:.2f}%')
        
        return total_loss / len(self.train_loader), correct / total
    
    def evaluate(self):
        """Évaluation du modèle"""
        self.model.eval()
        test_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                
                pred = output.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        test_loss /= len(self.test_loader)
        accuracy = accuracy_score(all_targets, all_preds)
        
        return test_loss, accuracy, all_preds, all_targets
    
    def train(self):
        """Boucle d'entraînement principale"""
        print("="*50)
        print("DÉMARRAGE DE L'ENTRAÎNEMENT MAMBA")
        print("="*50)
        
        best_acc = 0
        
        for epoch in range(self.args.epochs):
            print(f'\nÉpoque {epoch+1}/{self.args.epochs}')
            print("-" * 30)
            
            train_loss, train_acc = self.train_epoch()
            test_loss, test_acc, _, _ = self.evaluate()
            
            self.scheduler.step()
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
            print(f'LR: {self.scheduler.get_last_lr()[0]:.6f}')
            
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(self.model.state_dict(), 'best_mamba_model.pth')
                print(f'Nouveau meilleur modèle sauvegardé! Acc: {best_acc:.4f}')
    
    def test(self):
        """Test final du modèle"""
        print("="*50)
        print("ÉVALUATION FINALE")
        print("="*50)
        
        # Charger le meilleur modèle
        self.model.load_state_dict(torch.load('best_mamba_model.pth'))
        
        test_loss, test_acc, preds, targets = self.evaluate()
        
        # Matrice de confusion
        cm = confusion_matrix(targets, preds)
        tn, fp, fn, tp = cm.ravel()
        
        # Métriques
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)
        
        print(f"Accuracy: {test_acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1_score:.4f}")
        print(f"False Positive Rate: {fpr:.4f}")
        print(f"False Negative Rate: {fnr:.4f}")
        
        print("\nMatrice de confusion:")
        print(cm)
        
        print("\nRapport de classification:")
        print(classification_report(targets, preds, target_names=['Non-vulnérable', 'Vulnérable']))
        
        return {
            'accuracy': test_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'fpr': fpr,
            'fnr': fnr
        }