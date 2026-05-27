#!/usr/bin/env python3
"""
Convert REFIT dataset to XES format for ProM process mining.
"""

import xml.etree.ElementTree as ET
import csv
from datetime import datetime
from collections import defaultdict
import sys

def parse_building_metadata(xml_file, target_building_id=None):
    """Parse the REFIT building survey XML to extract sensor metadata."""
    print(f"Parsing building metadata from {xml_file}...")
    if target_building_id:
        print(f"  Filtering for building: {target_building_id}")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    ns = {'refit': 'http://www.refitsmarthomes.org'}
    
    # Dictionary to map TimeSeriesVariable ID to metadata
    variable_metadata = {}
    
    # Extract all TimeSeriesVariables with their context
    for building in root.findall('.//refit:Building', ns):
        building_id = building.get('id')
        
        # Skip if we're filtering for a specific building
        if target_building_id and building_id != target_building_id:
            continue
        
        # Get spaces (rooms) in the building
        for space in building.findall('.//refit:Space', ns):
            space_id = space.get('id')
            room_type = space.get('roomType', 'Unknown')
            storey_level = space.get('storeyLevel', '0')
            
            # Get sensors in this space
            for sensor in space.findall('.//refit:Sensor', ns):
                sensor_id = sensor.get('id')
                
                for var in sensor.findall('refit:TimeSeriesVariable', ns):
                    var_id = var.get('id')
                    var_type = var.get('variableType', 'Unknown')
                    units = var.get('units', '')
                    
                    variable_metadata[var_id] = {
                        'building': building_id,
                        'space': space_id,
                        'room_type': room_type,
                        'storey': storey_level,
                        'sensor': sensor_id,
                        'variable_type': var_type,
                        'units': units,
                        'context': f"{room_type}_{var_type}"
                    }
        
        # Get meters (electricity, gas)
        for meter in building.findall('.//refit:Meter', ns):
            meter_type = meter.get('meterType', 'Unknown')
            
            for sensor in meter.findall('.//refit:Sensor', ns):
                sensor_id = sensor.get('id')
                
                for var in sensor.findall('refit:TimeSeriesVariable', ns):
                    var_id = var.get('id')
                    var_type = var.get('variableType', 'Unknown')
                    units = var.get('units', '')
                    
                    variable_metadata[var_id] = {
                        'building': building_id,
                        'space': meter_type + '_Meter',
                        'room_type': meter_type + '_Meter',
                        'storey': '0',
                        'sensor': sensor_id,
                        'variable_type': var_type,
                        'units': units,
                        'context': f"{meter_type}_{var_type}"
                    }
    
    print(f"Found {len(variable_metadata)} TimeSeriesVariables")
    return variable_metadata


def select_diverse_variables(metadata, max_vars=15):
    """Select a diverse set of variables for the XES file."""
    # Group by variable type
    by_type = defaultdict(list)
    for var_id, meta in metadata.items():
        by_type[meta['variable_type']].append((var_id, meta))
    
    selected = []
    selected_types = []
    
    # Priority types for smart home analysis
    priority_types = [
        'Electrical power',      # Overall electricity usage
        'Air temperature',       # Temperature in rooms
        'Motion',               # Motion detection
        'Brightness',           # Light levels
        'Gas volume',           # Gas usage
        'Relative humidity'     # Humidity
    ]
    
    # First, get priority types
    for ptype in priority_types:
        if ptype in by_type and len(selected) < max_vars:
            # Take first few of each type from different rooms
            vars_of_type = by_type[ptype]
            # Get unique room types
            rooms_seen = set()
            for var_id, meta in vars_of_type:
                if meta['room_type'] not in rooms_seen or ptype in ['Electrical power', 'Gas volume']:
                    selected.append((var_id, meta))
                    rooms_seen.add(meta['room_type'])
                    selected_types.append(ptype)
                    if len(selected) >= max_vars:
                        break
    
    # Fill remaining slots with other interesting types
    for vtype, vars_list in by_type.items():
        if vtype not in priority_types and len(selected) < max_vars:
            selected.append(vars_list[0])
            selected_types.append(vtype)
    
    print(f"\nSelected {len(selected)} diverse variables:")
    for var_id, meta in selected:
        print(f"  - {var_id}: {meta['context']} ({meta['variable_type']}, {meta['units']})")
    
    return {var_id: meta for var_id, meta in selected}


def read_time_series_data(csv_file, selected_vars, start_date=None, end_date=None, max_events=5000):
    """Read time series data for selected variables within date range."""
    print(f"\nReading time series data from {csv_file}...")
    
    events = []
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for i, row in enumerate(reader):
            if i % 100000 == 0:
                print(f"  Processed {i} rows, collected {len(events)} events...")
            
            var_id = row['TimeSeriesVariable/@id']
            
            if var_id not in selected_vars:
                continue
            
            dt_str = row['dateTime']
            value = row['data']
            
            # Parse datetime
            dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
            
            # Filter by date range if specified
            if start_date and dt < start_date:
                continue
            if end_date and dt > end_date:
                continue
            
            meta = selected_vars[var_id]
            
            events.append({
                'timestamp': dt,
                'activity': meta['context'],
                'variable_id': var_id,
                'variable_type': meta['variable_type'],
                'value': value,
                'units': meta['units'],
                'building': meta['building'],
                'room': meta['room_type'],
                'space': meta['space'],
                'storey': meta['storey']
            })
            
            if len(events) >= max_events:
                print(f"  Reached maximum events limit ({max_events})")
                break
    
    # Sort events by timestamp
    events.sort(key=lambda x: x['timestamp'])
    
    print(f"Collected {len(events)} events")
    return events


def create_cases_from_events(events, time_window_hours=24):
    """Group events into cases based on time windows (e.g., daily routines)."""
    if not events:
        return []
    
    cases = []
    current_case = []
    case_start = events[0]['timestamp']
    case_id = 1
    
    for event in events:
        # Check if event is within current time window
        time_diff = (event['timestamp'] - case_start).total_seconds() / 3600
        
        if time_diff <= time_window_hours:
            current_case.append(event)
        else:
            # Start new case
            if current_case:
                cases.append((case_id, current_case))
                case_id += 1
            current_case = [event]
            case_start = event['timestamp']
    
    # Add last case
    if current_case:
        cases.append((case_id, current_case))
    
    print(f"\nCreated {len(cases)} cases with {time_window_hours}-hour time windows")
    return cases


def write_xes_file(cases, output_file, dataset_name="REFIT Smart Home"):
    """Write events to XES file format for ProM."""
    print(f"\nWriting XES file to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write XES header
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<log xes.version="1.0" xes.features="nested-attributes" openxes.version="1.0RC7">\n')
        
        # Write log-level attributes
        f.write(f'  <string key="concept:name" value="{dataset_name}"/>\n')
        f.write('  <string key="lifecycle:model" value="standard"/>\n')
        f.write('  <string key="source" value="REFIT Dataset"/>\n')
        
        # Write global event attributes (classifier)
        f.write('  <global scope="event">\n')
        f.write('    <string key="concept:name" value="__INVALID__"/>\n')
        f.write('    <date key="time:timestamp" value="1970-01-01T00:00:00.000+00:00"/>\n')
        f.write('    <string key="Activity" value="__INVALID__"/>\n')
        f.write('    <string key="org:resource" value="__INVALID__"/>\n')
        f.write('  </global>\n')
        
        # Write classifier
        f.write('  <classifier name="Activity" keys="Activity"/>\n')
        f.write('  <classifier name="activity classifier" keys="Activity"/>\n')
        
        # Write traces (cases)
        for case_id, events in cases:
            f.write('  <trace>\n')
            f.write(f'    <string key="concept:name" value="Case_{case_id}"/>\n')
            
            # Write events in the trace
            for event in events:
                f.write('    <event>\n')
                
                # Activity name (required)
                f.write(f'      <string key="concept:name" value="{escape_xml(event["activity"])}"/>\n')
                f.write(f'      <string key="Activity" value="{escape_xml(event["activity"])}"/>\n')
                
                # Timestamp (required)
                timestamp_str = event['timestamp'].strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + '+00:00'
                f.write(f'      <date key="time:timestamp" value="{timestamp_str}"/>\n')
                
                # Additional attributes
                f.write(f'      <string key="org:resource" value="{escape_xml(event["room"])}"/>\n')
                f.write(f'      <string key="Variable_Type" value="{escape_xml(event["variable_type"])}"/>\n')
                f.write(f'      <string key="Value" value="{escape_xml(event["value"])}"/>\n')
                f.write(f'      <string key="Units" value="{escape_xml(event["units"])}"/>\n')
                f.write(f'      <string key="Building" value="{escape_xml(event["building"])}"/>\n')
                f.write(f'      <string key="Room" value="{escape_xml(event["room"])}"/>\n')
                f.write(f'      <string key="Space_ID" value="{escape_xml(event["space"])}"/>\n')
                f.write(f'      <string key="Storey" value="{event["storey"]}"/>\n')
                f.write(f'      <string key="lifecycle:transition" value="complete"/>\n')
                
                f.write('    </event>\n')
            
            f.write('  </trace>\n')
        
        f.write('</log>\n')
    
    print(f"XES file created successfully: {output_file}")


def escape_xml(text):
    """Escape special XML characters."""
    if text is None:
        return ""
    text = str(text)
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    text = text.replace('"', '&quot;')
    text = text.replace("'", '&apos;')
    return text


def find_available_variables(csv_file, metadata, target_building_id=None, max_scan=500000):
    """Scan CSV to find which variables actually have data."""
    print(f"\nScanning CSV to find available variables (first {max_scan} rows)...")
    if target_building_id:
        print(f"  Looking specifically for {target_building_id} variables...")
    available_vars = set()
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_scan:
                break
            if i % 100000 == 0:
                print(f"  Scanned {i} rows, found {len(available_vars)} variables so far...")
            var_id = row['TimeSeriesVariable/@id']
            if var_id in metadata:
                # If filtering by building, check building matches
                if target_building_id is None or metadata[var_id]['building'] == target_building_id:
                    available_vars.add(var_id)
    
    print(f"Found {len(available_vars)} available variables with metadata")
    
    # Show what we found
    var_types = {}
    for var_id in available_vars:
        vtype = metadata[var_id]['variable_type']
        if vtype not in var_types:
            var_types[vtype] = []
        var_types[vtype].append(var_id)
    
    print("\nVariable types found:")
    for vtype, vars in sorted(var_types.items()):
        print(f"  {vtype}: {len(vars)} variables")
    
    return available_vars


def main():
    # Configuration
    target_building = 'Building02'  # Options: Building01, Building02, Building03...
    output_suffix = 'building02'     # Will create refit_building02.xes
    start_month = 10  # October
    start_day = 2
    end_month = 11  # November
    end_day = 2
    
    # File paths
    xml_file = r'c:\Users\Teore\Desktop\MSc\Datasets\Refit\REFIT_BUILDING_SURVEY.xml'
    csv_file = r'c:\Users\Teore\Desktop\MSc\Datasets\Refit\REFIT_TIME_SERIES_VALUES.csv'
    output_file = rf'c:\Users\Teore\Desktop\MSc\Datasets\Refit\refit_{output_suffix}.xes'
    
    print("=" * 70)
    print(f"REFIT to XES Converter - {target_building}")
    print("=" * 70)
    
    # Parse building metadata
    all_metadata = parse_building_metadata(xml_file, target_building_id=None)
    
    # Find which variables are actually in the CSV (with building filter)
    available_vars = find_available_variables(csv_file, all_metadata, target_building_id=target_building, max_scan=5000000)
    
    # Filter metadata to only available variables
    filtered_metadata = {k: v for k, v in all_metadata.items() if k in available_vars}
    print(f"Using {len(filtered_metadata)} variables that have both metadata and data")
    
    # Select variables
    selected_vars = select_diverse_variables(filtered_metadata, max_vars=20)
    
    # Read time series data frim specified month
    from datetime import timezone
    start_date = datetime(2013, start_month, start_day, 0, 0, 0, tzinfo=timezone.utc)
    end_date = datetime(2013, end_month, end_day, 23, 59, 59, tzinfo=timezone.utc)
    
    events = read_time_series_data(
        csv_file, 
        selected_vars, 
        start_date=start_date, 
        end_date=end_date,
        max_events=8000  # Limit to manageable size for ProM
    )
    
    if not events:
        print("ERROR: No events found! Check your date range and variable selection.")
        sys.exit(1)
    
    # Create cases
    cases = create_cases_from_events(events, time_window_hours=24)
    
    # Write XES file
    write_xes_file(cases, output_file, dataset_name=f"REFIT Smart Home - {target_building}")
    
    print("\n" + "=" * 70)
    print("Conversion complete!")
    print(f"Building: {target_building}")
    print(f"Time period: {start_month}/{start_day} - {end_month}/{end_day}, 2013")
    print(f"Total events: {len(events)}")
    print(f"Total cases: {len(cases)}")
    print(f"Output file: {output_file}")
    print("=" * 70)


if __name__ == '__main__':
    main()
