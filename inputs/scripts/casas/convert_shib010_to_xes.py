"""
Convert CASAS shib010.txt to XES format

"""

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom


def parse_shib010_txt(filepath):
    """Parse shib010.txt file into DataFrame."""
    
    print(f"Reading {filepath}...")
    
    # Read file manually to handle format properly
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                data.append(parts[:4])
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=['timestamp', 'sensor_id', 'value', 'sensor_type'])
    
    print(f"  Total events read: {len(df)}")
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    
    # Remove rows with invalid timestamps
    df = df.dropna(subset=['timestamp'])
    print(f"  Events with valid timestamps: {len(df)}")
    
    # Filter to December 2015 onwards (remove June data)
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Keep only December 2015 data
    df = df[(df['timestamp'] >= pd.Timestamp('2015-12-02')) & (df['timestamp'] < pd.Timestamp('2016-01-01'))]
    print(f"  Events after Dec 2015 filter: {len(df)}")
    
    # Ensure sensor_type and sensor_id are strings
    df['sensor_type'] = df['sensor_type'].astype(str)
    df['sensor_id'] = df['sensor_id'].astype(str)
    df['value'] = df['value'].astype(str)
    
    # Remove system events and errors
    df = df[~df['sensor_id'].isin(['system', 'Agent Started'])]
    df = df[~df['sensor_type'].str.contains('Radio_error', na=False)]
    df = df[~df['sensor_type'].str.contains('Zigbee-', na=False)]  # Remove Zigbee metadata
    
    # Remove Control4-Radio heartbeats (value="OK")
    radio_heartbeats = (df['sensor_type'] == 'Control4-Radio') & (df['value'] == 'OK')
    df = df[~radio_heartbeats]
    
    # Remove battery status events (not behavioral events)
    battery_events = df['sensor_type'].isin(['Control4-BatteryPercent', 'Control4-BatteryVoltage'])
    battery_count = battery_events.sum()
    df = df[~battery_events]
    print(f"  Filtered {battery_count} battery status events")
    
    print(f"  Events after filtering system/errors/heartbeats/battery: {len(df)}")
    
    return df


def categorize_light_level(value_str):
    """
    Convert raw light sensor value to meaningful category.
    Returns None if not a light sensor or invalid value.
    """
    try:
        val = int(value_str)
        if val < 15:
            return "DARK"
        elif val < 40:
            return "DIM"
        elif val < 70:
            return "MEDIUM"
        else:
            return "BRIGHT"
    except (ValueError, TypeError):
        return None


def preprocess_light_sensors(df):
    """
    Preprocess light sensor events to reduce noise.
    Only keep events where light state changes significantly.
    """
    print("\nPreprocessing light sensor events...")
    
    # Identify light sensor rows
    light_mask = df['sensor_type'] == 'Control4-LightSensor'
    
    if not light_mask.any():
        print("  No light sensor events found")
        return df
    
    # Add light category column
    df['light_category'] = None
    df.loc[light_mask, 'light_category'] = df.loc[light_mask, 'value'].apply(categorize_light_level)
    
    # Track previous state per sensor
    light_events_to_keep = []
    prev_states = {}
    
    for idx, row in df.iterrows():
        if row['sensor_type'] == 'Control4-LightSensor':
            sensor_id = row['sensor_id']
            current_state = row['light_category']
            
            # Keep if state changed OR it's first reading
            if sensor_id not in prev_states or prev_states[sensor_id] != current_state:
                light_events_to_keep.append(idx)
                prev_states[sensor_id] = current_state
        else:
            # Keep all non-light sensor events
            light_events_to_keep.append(idx)
    
    # Filter dataframe
    original_count = len(df)
    df = df.loc[light_events_to_keep].reset_index(drop=True)
    light_filtered = original_count - len(df)
    
    print(f"  Filtered {light_filtered} redundant light sensor events")
    print(f"  Kept {len(df)} events with meaningful light state changes")
    
    return df


def preprocess_temperature_sensors(df, threshold=1.0):
    """
    Preprocess temperature sensor events to reduce noise.
    Only keep events where temperature changes by more than threshold (degrees).
    """
    print(f"\nPreprocessing temperature sensor events (threshold={threshold}°C)...")
    
    # Identify temperature sensor rows
    temp_mask = df['sensor_type'] == 'Control4-Temperature'
    
    if not temp_mask.any():
        print("  No temperature sensor events found")
        return df
    
    # Track previous temperature per sensor
    temp_events_to_keep = []
    prev_temps = {}
    
    for idx, row in df.iterrows():
        if row['sensor_type'] == 'Control4-Temperature':
            sensor_id = row['sensor_id']
            try:
                current_temp = float(row['value'])
            except (ValueError, TypeError):
                continue  # Skip invalid temperature values
            
            # Keep if temperature changed significantly OR it's first reading
            if sensor_id not in prev_temps:
                temp_events_to_keep.append(idx)
                prev_temps[sensor_id] = current_temp
            else:
                temp_diff = abs(current_temp - prev_temps[sensor_id])
                if temp_diff >= threshold:
                    temp_events_to_keep.append(idx)
                    prev_temps[sensor_id] = current_temp
        else:
            # Keep all non-temperature sensor events
            temp_events_to_keep.append(idx)
    
    # Filter dataframe
    original_count = len(df)
    df = df.loc[temp_events_to_keep].reset_index(drop=True)
    temp_filtered = original_count - len(df)
    
    print(f"  Filtered {temp_filtered} redundant temperature events (< {threshold}°C change)")
    print(f"  Kept {len(df)} events with meaningful temperature changes")
    
    return df


def create_time_based_cases(df, window_hours=1):
    """
    Create case IDs based on time windows.
    Each case = activities within a time window.
    """
    
    print(f"\nCreating time-based cases (window={window_hours}h)...")
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Create case IDs based on time windows
    start_time = df['timestamp'].min()
    df['case_id'] = ((df['timestamp'] - start_time).dt.total_seconds() / (window_hours * 3600)).astype(int)
    df['case_id'] = 'case_' + df['case_id'].astype(str)
    
    print(f"  Created {df['case_id'].nunique()} cases")
    print(f"  Average events per case: {len(df) / df['case_id'].nunique():.1f}")
    
    return df


def map_sensors_to_activities(df):
    """
    Map low-level sensor IDs to location-based activities.
    Creates meaningful activity names for process mining.
    """
    
    print("\nMapping sensors to location-based activities...")
    
    def create_activity(row):
        """
        Create activity name from sensor location and state.
        Format: Location_Item_State (e.g., "BedroomA_Light_BRIGHT", "Kitchen_Sink")
        """
        sensor = str(row['sensor_id'])
        sensor_type = str(row['sensor_type'])
        
        # Parse location from sensor ID
        # Examples: BedroomABed, KitchenASink, BathroomAToilet, MainEntryway
        
        # Special handling for LIGHT SENSORS
        if sensor_type == 'Control4-LightSensor':
            light_state = row.get('light_category', 'UNKNOWN')
            # Clean up sensor name
            import re
            formatted = re.sub(r'([A-Z])', r'_\1', sensor).lstrip('_')
            return f"{formatted}_Light_{light_state}"
        
        # Handle other special cases
        elif 'Relay' in sensor:
            # Temperature relay sensors
            if 'Temperature' in sensor_type:
                return f"Temperature_{sensor}"
            else:
                return f"Control_{sensor}"
        
        # Motion sensors (MainEntryway, MainDoor, etc.)
        elif 'Motion' in sensor_type or sensor in ['MainEntryway', 'MainDoor']:
            # Clean up name: MainEntryway → Main_Entryway
            if 'Main' in sensor:
                return sensor.replace('Main', 'Main_')
            return f"Motion_{sensor}"
        
        # Location-based sensors (BedroomABed, KitchenASink, etc.)
        else:
            # Parse: BedroomABed → BedroomA_Bed
            # Split on capital letters to identify components
            import re
            
            # Insert underscores before capitals (except first)
            formatted = re.sub(r'([A-Z])', r'_\1', sensor).lstrip('_')
            
            # Clean up common patterns
            formatted = formatted.replace('Room_A_', 'RoomA_')
            formatted = formatted.replace('Area_', '')  # Remove redundant "Area"
            
            return formatted
    
    df['activity'] = df.apply(create_activity, axis=1)
    
    # Clean up activity names further
    df['activity'] = df['activity'].str.replace('__', '_')
    
    print(f"  Mapped to {df['activity'].nunique()} unique activities")
    
    # Show light sensor statistics if present
    light_activities = df[df['sensor_type'] == 'Control4-LightSensor']
    if not light_activities.empty:
        print(f"\n  Light sensor activities (top 10):")
        for act, count in light_activities['activity'].value_counts().head(10).items():
            print(f"    - {act}: {count}")
    
    print(f"\n  All activities (top 20):")
    for act, count in df['activity'].value_counts().head(20).items():
        print(f"    - {act}: {count}")
    
    return df


def create_xes(df, output_path):
    """Create XES XML structure from DataFrame."""
    
    print(f"\nCreating XES file: {output_path}")
    
    # Create root element
    log = ET.Element('log')
    log.set('xes.version', '1.0')
    log.set('xes.features', 'nested-attributes')
    log.set('xmlns', 'http://www.xes-standard.org/')
    
    # Add extensions
    extension = ET.SubElement(log, 'extension')
    extension.set('name', 'Concept')
    extension.set('prefix', 'concept')
    extension.set('uri', 'http://www.xes-standard.org/concept.xesext')
    
    extension = ET.SubElement(log, 'extension')
    extension.set('name', 'Time')
    extension.set('prefix', 'time')
    extension.set('uri', 'http://www.xes-standard.org/time.xesext')
    
    extension = ET.SubElement(log, 'extension')
    extension.set('name', 'Organizational')
    extension.set('prefix', 'org')
    extension.set('uri', 'http://www.xes-standard.org/org.xesext')
    
    # Add global trace attributes
    global_trace = ET.SubElement(log, 'global')
    global_trace.set('scope', 'trace')
    
    string_attr = ET.SubElement(global_trace, 'string')
    string_attr.set('key', 'concept:name')
    string_attr.set('value', '__INVALID__')
    
    # Add global event attributes
    global_event = ET.SubElement(log, 'global')
    global_event.set('scope', 'event')
    
    string_attr = ET.SubElement(global_event, 'string')
    string_attr.set('key', 'concept:name')
    string_attr.set('value', '__INVALID__')
    
    date_attr = ET.SubElement(global_event, 'date')
    date_attr.set('key', 'time:timestamp')
    date_attr.set('value', '1970-01-01T00:00:00.000+00:00')
    
    string_attr = ET.SubElement(global_event, 'string')
    string_attr.set('key', 'org:resource')
    string_attr.set('value', '__INVALID__')
    
    # Add classifier
    classifier = ET.SubElement(log, 'classifier')
    classifier.set('name', 'Activity')
    classifier.set('keys', 'concept:name')
    
    # Group events by case
    cases = df.groupby('case_id')
    
    print(f"  Writing {len(cases)} traces...")
    
    for case_id, case_events in cases:
        trace = ET.SubElement(log, 'trace')
        
        # Trace name
        trace_name = ET.SubElement(trace, 'string')
        trace_name.set('key', 'concept:name')
        trace_name.set('value', str(case_id))
        
        # Add events to trace
        for _, event_row in case_events.iterrows():
            event = ET.SubElement(trace, 'event')
            
            # Activity name
            activity = ET.SubElement(event, 'string')
            activity.set('key', 'concept:name')
            activity.set('value', str(event_row['activity']))
            
            # Timestamp
            timestamp = ET.SubElement(event, 'date')
            timestamp.set('key', 'time:timestamp')
            timestamp.set('value', event_row['timestamp'].isoformat())
            
            # Resource (sensor ID)
            resource = ET.SubElement(event, 'string')
            resource.set('key', 'org:resource')
            resource.set('value', str(event_row['sensor_id']))
            
            # Sensor type
            sensor_type = ET.SubElement(event, 'string')
            sensor_type.set('key', 'sensor:type')
            sensor_type.set('value', str(event_row['sensor_type']))
            
            # Sensor value
            sensor_value = ET.SubElement(event, 'string')
            sensor_value.set('key', 'sensor:value')
            sensor_value.set('value', str(event_row['value']))
    
    # Pretty print and save
    xml_str = ET.tostring(log, encoding='utf-8')
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent='  ')
    
    # Remove extra blank lines
    pretty_xml = '\n'.join([line for line in pretty_xml.split('\n') if line.strip()])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)
    
    print(f"  XES file created successfully!")
    print(f"  File size: {Path(output_path).stat().st_size / 1024:.1f} KB")


def main():
    """Main conversion pipeline."""
    
    print("="*80)
    print("CASAS shib010.txt to XES Conversion (with Light Sensor Abstraction)")
    print("="*80)
    
    # File paths
    input_file = Path('shib010.txt')
    output_file = Path('shib010.xes')
    
    # Step 1: Parse input file
    df = parse_shib010_txt(input_file)
    
    # Step 2: Preprocess light sensors
    df = preprocess_light_sensors(df)
    
    # Step 3: Preprocess temperature sensors
    df = preprocess_temperature_sensors(df, threshold=1.0)
    
    # Step 4: Create time-based cases
    df = create_time_based_cases(df, window_hours=1)
    
    # Step 5: Map sensors to activities (with light abstraction)
    df = map_sensors_to_activities(df)
    
    # Step 5: Create XES file
    create_xes(df, output_file)
    
    print("\n" + "="*80)
    print("Conversion Complete!")
    print("="*80)
    print(f"\nOutput file: {output_file.absolute()}")
    print(f"Total traces: {df['case_id'].nunique()}")
    print(f"Total events: {len(df)}")
    print(f"Unique activities: {df['activity'].nunique()}")
    
    # Show light sensor statistics
    light_events = df[df['sensor_type'] == 'Control4-LightSensor']
    if not light_events.empty:
        print(f"\nLight sensor events: {len(light_events)}")
        print(f"Light state distribution:")
        for state, count in light_events['light_category'].value_counts().items():
            print(f"  {state}: {count}")
    


if __name__ == '__main__':
    main()
