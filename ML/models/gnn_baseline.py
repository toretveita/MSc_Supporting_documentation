"""Graph-temporal (Temporal GNN) baseline model.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np
from typing import Tuple, Dict
from collections import defaultdict


class BaselineGNN(nn.Module):
    """
    Graph-temporal GNN using data-driven graph construction.
    """
    
    def __init__(
        self,
        num_activities: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_gcn_layers: int = 2,
        lstm_layers: int = 2,
        dropout: float = 0.3
    ):

        super(BaselineGNN, self).__init__()
        
        self.num_activities = num_activities
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Activity embedding
        self.activity_embedding = nn.Embedding(num_activities, embedding_dim)
        
        # Additional feature projection
        self.feature_projection = nn.Linear(3, embedding_dim // 2)
        
        # Graph Convolutional Network layers
        # These learn activity relationships from temporal co-occurrence
        self.gcn_layers = nn.ModuleList()
        
        # First GCN layer
        self.gcn_layers.append(
            GCNConv(
                in_channels=embedding_dim + embedding_dim // 2,
                out_channels=hidden_dim
            )
        )
        
        # Additional GCN layers
        for _ in range(num_gcn_layers - 1):
            self.gcn_layers.append(
                GCNConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim
                )
            )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_activities)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        activity_sequences: torch.Tensor,
        additional_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass using data-driven graph structure.

        """
        batch_size, seq_length = activity_sequences.shape
        
        # === STEP 1: Create graph-enhanced activity representations ===
        # Uses an empirical co-occurrence graph from the event log
        
        activity_node_features = self.activity_embedding.weight
        
        # Add dummy features
        dummy_features = torch.zeros(
            self.num_activities,
            self.embedding_dim // 2,
            device=activity_node_features.device
        )
        x = torch.cat([activity_node_features, dummy_features], dim=-1)
        
        # Apply GCN layers
        for gcn_layer in self.gcn_layers:
            x = gcn_layer(x, edge_index, edge_weight=edge_weights)
            x = F.relu(x)
            x = self.dropout(x)
        
        graph_enhanced_embeddings = x
        
        # === STEP 2: Process sequences ===
        sequence_embeddings = graph_enhanced_embeddings[activity_sequences]
        
        # === STEP 3: LSTM temporal modeling ===
        lstm_out, (hidden, cell) = self.lstm(sequence_embeddings)
        last_hidden = lstm_out[:, -1, :]
        
        # === STEP 4: Prediction ===
        x = self.layer_norm(last_hidden)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits
    
    def predict_next_activity(
        self,
        activity_sequences: torch.Tensor,
        additional_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next activity with probabilities.
        """
        logits = self.forward(activity_sequences, additional_features, edge_index, edge_weights)
        probabilities = F.softmax(logits, dim=-1)
        predicted_activities = torch.argmax(probabilities, dim=-1)
        
        return predicted_activities, probabilities


class DataDrivenGraphBuilder:
    """
    Constructs activity graph from event log based on temporal co-occurrence.
    """
    
    def __init__(self, min_support: float = 0.01):
        """
        Args:
            min_support: Minimum relative frequency for edge creation
        """
        self.min_support = min_support
        self.co_occurrence_matrix = None
        
    def build_graph_from_sequences(
        self,
        X: np.ndarray,
        num_activities: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build activity graph from event sequences.

        """
        print("\nBuilding data-driven activity graph...")
        
        # Count co-occurrences
        co_occurrence = defaultdict(int)
        total_transitions = 0
        
        for sequence in X:
            for i in range(len(sequence) - 1):
                act1, act2 = sequence[i], sequence[i + 1]
                co_occurrence[(act1, act2)] += 1
                total_transitions += 1
        
        print(f"Total activity transitions: {total_transitions}")
        print(f"Unique activity pairs: {len(co_occurrence)}")
        
        # Create edges with minimum support threshold
        edges = []
        weights = []
        
        min_count = int(total_transitions * self.min_support)
        
        for (act1, act2), count in co_occurrence.items():
            if count >= min_count:
                edges.append([act1, act2])
                # Normalize weight
                weight = count / total_transitions
                weights.append(weight)
        
        # Add self-loops for all activities
        for i in range(num_activities):
            edges.append([i, i])
            weights.append(0.1)  # Small self-loop weight
        
        edge_index = np.array(edges).T if edges else np.zeros((2, 0))
        edge_weights = np.array(weights) if weights else np.array([])
        
        # Normalize weights
        if len(edge_weights) > 0:
            edge_weights = edge_weights / edge_weights.sum()
        
        print(f"Created graph with {edge_index.shape[1]} edges")
        print(f"Average out-degree: {edge_index.shape[1] / num_activities:.2f}")
        
        return edge_index, edge_weights


class BaselineGNNTrainer:
    """Trainer for baseline GNN model."""
    
    def __init__(
        self,
        model: BaselineGNN,
        edge_index: np.ndarray,
        edge_weights: np.ndarray = None,
        learning_rate: float = 0.001,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        
        # Convert graph structure to tensors
        self.edge_index = torch.tensor(edge_index, dtype=torch.long, device=device)
        self.edge_weights = torch.tensor(edge_weights, dtype=torch.float, device=device) if edge_weights is not None else None
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, X_train: np.ndarray, y_train: np.ndarray, batch_size: int = 32) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        indices = np.random.permutation(len(X_train))
        
        for i in range(0, len(X_train), batch_size):
            batch_indices = indices[i:i+batch_size]
            
            batch_X = torch.tensor(X_train[batch_indices], dtype=torch.long, device=self.device)
            batch_y = torch.tensor(y_train[batch_indices], dtype=torch.long, device=self.device)
            
            batch_features = torch.zeros(batch_X.shape[0], batch_X.shape[1], 3, device=self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(batch_X, batch_features, self.edge_index, self.edge_weights)
            loss = self.criterion(logits, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches

    def evaluate(self, test_sequences: np.ndarray, test_labels: np.ndarray, batch_size: int = 32) -> Tuple[float, float]:
        """Evaluate on test set."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        num_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(test_sequences), batch_size):
                batch_X = torch.tensor(test_sequences[i:i+batch_size], dtype=torch.long, device=self.device)
                batch_y = torch.tensor(test_labels[i:i+batch_size], dtype=torch.long, device=self.device)
                
                batch_features = torch.zeros(batch_X.shape[0], batch_X.shape[1], 3, device=self.device)
                
                logits = self.model(batch_X, batch_features, self.edge_index, self.edge_weights)
                loss = self.criterion(logits, batch_y)
                
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == batch_y).sum().item()
                total += batch_y.size(0)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        accuracy = correct / total
        
        return avg_loss, accuracy

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
        """
        Training loop with evaluation.
        

        """
        train_losses = []
        test_losses = []
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_sequences, train_labels, batch_size)
            test_metrics = self.evaluate(test_sequences, test_labels, batch_size)
            test_acc = test_metrics['accuracy']
            test_loss = test_metrics.get('loss', 0.0)
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        return train_losses, test_losses

    def evaluate(self, test_sequences: np.ndarray, test_labels: np.ndarray, batch_size: int = 32) -> dict:
        """
        Evaluate model and return comprehensive metrics.
        
        Returns:
            Dictionary with accuracy, top-k metrics, predictions, probabilities
        """
        self.model.eval()
        
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for i in range(0, len(test_sequences), batch_size):
                batch_X = torch.tensor(test_sequences[i:i+batch_size], dtype=torch.long, device=self.device)
                batch_features = torch.zeros(batch_X.shape[0], batch_X.shape[1], 3, device=self.device)
                
                logits = self.model(batch_X, batch_features, self.edge_index, self.edge_weights)
                probs = torch.softmax(logits, dim=1)
                
                all_predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
                all_probabilities.append(probs.cpu().numpy())
        
        predictions = np.concatenate(all_predictions)
        probabilities = np.concatenate(all_probabilities)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(test_labels, predictions)
        
        # Top-k accuracy
        def top_k_accuracy(y_true, y_probs, k):
            if y_probs.shape[1] < k:
                k = y_probs.shape[1]
            top_k_preds = np.argsort(y_probs, axis=1)[:, -k:]
            correct = sum(y in preds for y, preds in zip(y_true, top_k_preds))
            return correct / len(y_true)
        
        top_3 = top_k_accuracy(test_labels, probabilities, 3)
        top_5 = top_k_accuracy(test_labels, probabilities, 5)
        
        return {
            'accuracy': float(accuracy),
            'top_3_accuracy': float(top_3),
            'top_5_accuracy': float(top_5),
            'mean_confidence': float(np.mean(np.max(probabilities, axis=1))),
            'predictions': predictions,
            'probabilities': probabilities
        }
