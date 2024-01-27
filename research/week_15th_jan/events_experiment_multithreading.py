import multiprocessing
from itertools import repeat

import numpy as np
import torch


def analyze_row(row):
    in_event = False
    event_start = 0
    num_events = 0
    max_values = []
    avg_values = []
    durations = []

    for i, value in enumerate(row):
        value = value.item()
        if value > 0:
            if not in_event:
                in_event = True
                event_start = i
                num_events += 1
                max_value = value
                total_value = value
                start_position = i
            else:
                max_value = max(max_value, value)
                total_value += value
        else:
            if in_event:
                in_event = False
                durations.append(i - event_start)
                max_values.append(max_value)
                avg_values.append(total_value / (i - event_start))
                final_position = i

    if in_event:
        durations.append(len(row) - event_start)
        max_values.append(max_value)
        avg_values.append(total_value / (len(row) - event_start))
    
        # get the average event duration
        avg_duration = (sum(durations) / len(durations)) if len(durations) > 0 else np.NaN
        
        # max duration 
        max_duration = max(durations) if len(durations) > 0 else np.NaN
        
        # get the average max value
        avg_max_value = (sum(max_values) / (len(max_values)) if len(max_values) > 0 else np.NaN)
        num_firings = sum(durations)
        
        # `zip` avg_valuea, max_values, durations and add it as a subrecord which we could unfurl later
        event_stats = zip(avg_values, max_values, durations)
        event_stats = [
            {
                'avg_value': avg_value,
                'max_value': max_value,
                'duration': duration,
                'start_position': start_position, 
                'final_position': final_position,
            }
            for avg_value, max_value, duration in event_stats
        ]


        results = {
            'num_events': num_events,
            'num_firings': num_firings,
            'avg_values': avg_values,
            'max_values': max_values,
            'durations': durations,
            'avg_duration': avg_duration,
            'max_duration': max_duration,
            'avg_max_value': avg_max_value,
            'events': event_stats,
        }

        
        return results

def analyze_events_parallel(tensor, num_processes=None):
    
    
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    with multiprocessing.Pool(num_processes) as pool:
        results = pool.map(analyze_row, tensor)

    return results
