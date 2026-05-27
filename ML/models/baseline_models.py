"""
Baseline Models for Comparison
================================

"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')


class TraditionalMLBaselines:
    """Traditional machine learning baselines."""
    
    @staticmethod
    def train_random_forest(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_estimators: int = 100,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Random Forest baseline.
        
        """
        # Flatten sequences for traditional ML
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        print("  Training Random Forest...")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )
        model.fit(X_train_flat, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_flat)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get probabilities for top-k metrics
        y_proba = model.predict_proba(X_test_flat)
        
        # Top-k accuracy
        top_3_acc = TraditionalMLBaselines._top_k_accuracy(y_test, y_proba, k=3)
        top_5_acc = TraditionalMLBaselines._top_k_accuracy(y_test, y_proba, k=5)
        
        print(f"  ✓ Random Forest Accuracy: {accuracy:.4f}")
        
        return {
            'model_type': 'Random Forest',
            'accuracy': float(accuracy),
            'top_3_accuracy': float(top_3_acc),
            'top_5_accuracy': float(top_5_acc),
            'mean_confidence': float(np.mean(np.max(y_proba, axis=1))),
            'model': model,
            'predictions': y_pred
        }
    
    @staticmethod
    def train_gradient_boosting(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_estimators: int = 100,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Gradient Boosting baseline.
        
        """
        # Flatten sequences
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        print("  Training Gradient Boosting...")
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            verbose=0
        )
        model.fit(X_train_flat, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_flat)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get probabilities
        y_proba = model.predict_proba(X_test_flat)
        
        # Top-k accuracy
        top_3_acc = TraditionalMLBaselines._top_k_accuracy(y_test, y_proba, k=3)
        top_5_acc = TraditionalMLBaselines._top_k_accuracy(y_test, y_proba, k=5)
        
        print(f"  ✓ Gradient Boosting Accuracy: {accuracy:.4f}")
        
        return {
            'model_type': 'Gradient Boosting',
            'accuracy': float(accuracy),
            'top_3_accuracy': float(top_3_acc),
            'top_5_accuracy': float(top_5_acc),
            'mean_confidence': float(np.mean(np.max(y_proba, axis=1))),
            'model': model,
            'predictions': y_pred
        }
    
    @staticmethod
    def _top_k_accuracy(y_true: np.ndarray, y_proba: np.ndarray, k: int) -> float:
        """Calculate top-k accuracy."""
        top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]
        correct = sum(y in preds for y, preds in zip(y_true, top_k_preds))
        return correct / len(y_true)


class LSTMOnlyModel(nn.Module):
    """Pure LSTM baseline without any graph structure."""
    
    def __init__(
        self,
        num_activities: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(num_activities, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_activities)
    
    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            sequences: (batch_size, seq_len)
            
        Returns:
            logits: (batch_size, num_activities)
        """
        # Embed sequences
        embedded = self.embedding(sequences)  # (batch, seq_len, embedding_dim)
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)  # (batch, seq_len, hidden_dim)
        
        # Take last output
        last_output = lstm_out[:, -1, :]  # (batch, hidden_dim)
        
        # Dropout and classify
        dropped = self.dropout(last_output)
        logits = self.fc(dropped)  # (batch, num_activities)
        
        return logits


class LSTMOnlyTrainer:
    """Trainer for LSTM-only baseline."""
    
    def __init__(
        self,
        model: LSTMOnlyModel,
        device: torch.device = None,
        learning_rate: float = 0.001
    ):
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_losses = []
        self.test_losses = []
    
    def train(
        self,
        train_sequences: np.ndarray,
        train_labels: np.ndarray,
        test_sequences: np.ndarray,
        test_labels: np.ndarray,
        num_epochs: int = 30,
        batch_size: int = 32,
        verbose: bool = True
    ) -> Tuple[list, list]:
        """Train the model."""
        
        for epoch in range(num_epochs):
            self.model.train()
            
            # Shuffle training data
            indices = np.random.permutation(len(train_sequences))
            train_sequences_shuffled = train_sequences[indices]
            train_labels_shuffled = train_labels[indices]
            
            epoch_losses = []
            
            # Mini-batch training
            for i in range(0, len(train_sequences), batch_size):
                batch_sequences = train_sequences_shuffled[i:i+batch_size]
                batch_labels = train_labels_shuffled[i:i+batch_size]
                
                # Convert to tensors (sequences are already activity indices)
                sequences_tensor = torch.LongTensor(batch_sequences).to(self.device)
                labels_tensor = torch.LongTensor(batch_labels).to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                logits = self.model(sequences_tensor)
                loss = self.criterion(logits, labels_tensor)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_losses.append(loss.item())
            
            avg_train_loss = np.mean(epoch_losses)
            self.train_losses.append(avg_train_loss)
            
            # Evaluate on test set
            test_loss = self._compute_loss(test_sequences, test_labels, batch_size)
            self.test_losses.append(test_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}")
        
        return self.train_losses, self.test_losses
    
    def _compute_loss(self, sequences: np.ndarray, labels: np.ndarray, batch_size: int) -> float:
        """Compute loss on a dataset."""
        self.model.eval()
        losses = []
        
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch_sequences = sequences[i:i+batch_size]
                batch_labels = labels[i:i+batch_size]
                
                sequences_tensor = torch.LongTensor(batch_sequences).to(self.device)
                labels_tensor = torch.LongTensor(batch_labels).to(self.device)
                
                logits = self.model(sequences_tensor)
                loss = self.criterion(logits, labels_tensor)
                losses.append(loss.item())
        
        return np.mean(losses)
    
    def evaluate(self, test_sequences: np.ndarray, test_labels: np.ndarray) -> Dict[str, float]:
        """Evaluate the model."""
        self.model.eval()
        
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for i in range(0, len(test_sequences), 32):
                batch_sequences = test_sequences[i:i+32]
                
                sequences_tensor = torch.LongTensor(batch_sequences).to(self.device)
                logits = self.model(sequences_tensor)
                probs = torch.softmax(logits, dim=1)
                
                all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
                all_probs.append(probs.cpu().numpy())
        
        predictions = np.concatenate(all_preds)
        probabilities = np.concatenate(all_probs)
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predictions)
        
        # Top-k accuracy
        top_3_acc = self._top_k_accuracy(test_labels, probabilities, k=3)
        top_5_acc = self._top_k_accuracy(test_labels, probabilities, k=5)
        
        return {
            'accuracy': float(accuracy),
            'top_3_accuracy': float(top_3_acc),
            'top_5_accuracy': float(top_5_acc),
            'mean_confidence': float(np.mean(np.max(probabilities, axis=1))),
            'predictions': predictions,
            'probabilities': probabilities
        }
    
    def _top_k_accuracy(self, y_true: np.ndarray, y_proba: np.ndarray, k: int) -> float:
        """Calculate top-k accuracy."""
        if y_proba.shape[1] < k:
            k = y_proba.shape[1]
        top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]
        correct = sum(y in preds for y, preds in zip(y_true, top_k_preds))
        return correct / len(y_true)
