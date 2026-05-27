"""
Data Preprocessing for REFIT XES Event Log

"""

import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from collections import defaultdict


class XESParser:
    """Parser for XES event log files."""
    
    def __init__(self, xes_file_path: str):
        self.xes_file_path = xes_file_path
        self.activity_to_idx = {}
        self.resource_to_idx = {}
        self.idx_to_activity = {}
        
    def parse_event(self, event_elem) -> Dict:
        """Extract event attributes from XML element."""
        event_data = {}
        
        for child in event_elem:
            key = child.get('key')
            value = child.get('value')
            
            if key == 'Activity' or key == 'concept:name':
                event_data['activity'] = value
            elif key == 'time:timestamp':
                event_data['timestamp'] = datetime.fromisoformat(value.replace('+00:00', ''))
            elif key == 'org:resource':
                event_data['resource'] = value
            elif key == 'Value':
                try:
                    event_data['sensor_value'] = float(value)
                except (ValueError, TypeError):
                    event_data['sensor_value'] = 0.0 # Default for non-numeric values
                    event_data['sensor_value_text'] = value
            elif key == 'Variable_Type':
                event_data['variable_type'] = value
            elif key == 'Room':
                event_data['room'] = value
            elif key == 'Storey':
                event_data['storey'] = value
                
        return event_data
    
    def parse_trace(self, trace_elem) -> List[Dict]:
        """Extract all events from a trace (case)."""
        events = []
        case_id = None
        
        for child in trace_elem:
            tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            
            if tag == 'string' and child.get('key') == 'concept:name':
                case_id = child.get('value')
            elif tag == 'event':
                event_data = self.parse_event(child)
                if event_data and 'activity' in event_data:
                    event_data['case_id'] = case_id
                    events.append(event_data)
                    
        return events
    
    def parse_xes(self, max_traces: int = None) -> pd.DataFrame:
        """
        Parse XES file and return DataFrame with all events.
        
        """
        print(f"Parsing XES file: {self.xes_file_path}")
        
        all_events = []
        trace_count = 0
        
        context = ET.iterparse(self.xes_file_path, events=('start', 'end'))
        context = iter(context)
        
        for event, elem in context:
            tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            
            if event == 'end' and tag == 'trace':
                events = self.parse_trace(elem)
                all_events.extend(events)
                elem.clear()
                
                trace_count += 1
                if max_traces and trace_count >= max_traces:
                    break
                    
                if trace_count % 10 == 0:
                    print(f"Processed {trace_count} traces, {len(all_events)} events")
        
        print(f"Total traces parsed: {trace_count}")
        print(f"Total events parsed: {len(all_events)}")
        
        df = pd.DataFrame(all_events)
        
        # Create activity and resource indices
        unique_activities = df['activity'].unique()
        self.activity_to_idx = {act: idx for idx, act in enumerate(unique_activities)}
        self.idx_to_activity = {idx: act for act, idx in self.activity_to_idx.items()}
        
        unique_resources = df['resource'].unique()
        self.resource_to_idx = {res: idx for idx, res in enumerate(unique_resources)}
        
        print(f"Unique activities: {len(unique_activities)}")
        print(f"Unique resources: {len(unique_resources)}")
        
        return df
    
    def create_sequences(self, df: pd.DataFrame, sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray, List]:
        """
        Create sequences for next-event prediction.
        
        """
        sequences = []
        targets = []
        metadata = []
        
        # Group by case
        for case_id, group in df.groupby('case_id'):
            group = group.sort_values('timestamp').reset_index(drop=True)
            
            # Extract activity indices
            activities = [self.activity_to_idx[act] for act in group['activity']]
            
            # Create sliding windows
            for i in range(len(activities) - sequence_length):
                seq = activities[i:i+sequence_length]
                target = activities[i+sequence_length]
                
                sequences.append(seq)
                targets.append(target)
                
                # Extract metadata for this sequence
                seq_data = group.iloc[i:i+sequence_length]
                meta = {
                    'case_id': case_id,
                    'timestamps': seq_data['timestamp'].tolist(),
                    'sensor_values': seq_data['sensor_value'].tolist() if 'sensor_value' in seq_data else [],
                    'resources': [self.resource_to_idx[r] for r in seq_data['resource']],
                    'time_deltas': self._calculate_time_deltas(seq_data['timestamp'].tolist())
                }
                metadata.append(meta)
        
        X = np.array(sequences)
        y = np.array(targets)
        
        print(f"Created {len(sequences)} sequences")
        print(f"Input shape: {X.shape}, Target shape: {y.shape}")
        
        return X, y, metadata
    
    def _calculate_time_deltas(self, timestamps: List[datetime]) -> List[float]:
        """Calculate time differences between consecutive events in seconds."""
        if len(timestamps) < 2:
            return [0.0]
        
        deltas = []
        for i in range(1, len(timestamps)):
            delta = (timestamps[i] - timestamps[i-1]).total_seconds()
            deltas.append(delta)
        
        return deltas


def create_feature_matrix(X: np.ndarray, metadata: List[Dict], num_activities: int) -> np.ndarray:
    """
    Create feature matrix with temporal and value features.

    """
    num_sequences, seq_length = X.shape
    
    # Features: activity_embedding + time_delta + sensor_value + resource
    num_features = num_activities + 3  # one-hot activities + 3 additional features
    
    features = np.zeros((num_sequences, seq_length, num_features))
    
    for i in range(num_sequences):
        for j in range(seq_length):
            # One-hot encode activity
            activity_idx = X[i, j]
            features[i, j, activity_idx] = 1.0
            
            # Add time delta
            if j < len(metadata[i]['time_deltas']):
                time_delta = metadata[i]['time_deltas'][j]
                features[i, j, num_activities] = np.log1p(time_delta) / 10.0  # Log-scale normalization
            
            # Add sensor value
            if j < len(metadata[i]['sensor_values']):
                sensor_val = metadata[i]['sensor_values'][j]
                features[i, j, num_activities + 1] = sensor_val / 30.0  # Normalize temperature
            
            # Add resource index
            if j < len(metadata[i]['resources']):
                resource_idx = metadata[i]['resources'][j]
                features[i, j, num_activities + 2] = resource_idx / 10.0  # Normalize resource ID
    
    return features
