#!/usr/bin/env python3
"""
Interactive Log Viewer for LOVE_YOURSELF Event Logs

Usage:
    python debug/log_viewer.py                    # Browse all logs
    python debug/log_viewer.py --recent 10        # Show last 10 runs
    python debug/log_viewer.py --search "caption" # Search for specific events
    python debug/log_viewer.py --run abc123       # View specific run
"""

import json
import os
import sys
import argparse
import io

# Fix Windows console encoding for emojis
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import glob

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import MOOD_SNAPSHOT_FOLDER


class LogViewer:
    def __init__(self, log_dir: str = MOOD_SNAPSHOT_FOLDER):
        self.log_dir = log_dir
        self.logs = self._load_all_logs()
        
    def _load_all_logs(self) -> Dict[str, List]:
        """Load all event log files."""
        logs = {}
        pattern = os.path.join(self.log_dir, "*-event-log.json")
        
        for log_file in sorted(glob.glob(pattern)):
            run_id = os.path.basename(log_file).replace("-event-log.json", "")
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    logs[run_id] = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                continue
                
        return logs
    
    def list_runs(self, limit: int = 20) -> None:
        """List available runs with summary info."""
        print(f"\n{'='*80}")
        print(f"Found {len(self.logs)} runs in {self.log_dir}")
        print(f"{'='*80}\n")
        
        # Sort by most recent
        sorted_runs = sorted(self.logs.items(), 
                           key=lambda x: self._get_start_time(x[1]), 
                           reverse=True)[:limit]
        
        for run_id, events in sorted_runs:
            self._print_run_summary(run_id, events)
            
    def _get_start_time(self, events: List) -> float:
        """Get the start time of a run."""
        if events and 'timestamp' in events[0]:
            return events[0]['timestamp']
        return 0
    
    def _print_run_summary(self, run_id: str, events: List) -> None:
        """Print a summary of a single run."""
        if not events:
            return
            
        start_time = events[0].get('timestamp', 0)
        end_time = events[-1].get('timestamp', start_time)
        duration = end_time - start_time
        
        # Count event types
        event_types = {}
        captions = []
        reflections = []
        
        for event in events:
            event_type = event.get('type', 'unknown')
            event_types[event_type] = event_types.get(event_type, 0) + 1
            
            if event_type == 'caption':
                # Caption can be directly in event or in data field
                caption = event.get('caption', '') or event.get('data', {}).get('caption', '')
                if caption and not caption.startswith('['):
                    captions.append(caption[:50] + '...' if len(caption) > 50 else caption)
            elif event_type == 'reflection':
                # Reflection can be directly in event or in data field
                reflection = event.get('reflection', '') or event.get('data', {}).get('reflection', '')
                if reflection:
                    reflections.append(reflection[:50] + '...' if len(reflection) > 50 else reflection)
        
        print(f"Run ID: {run_id}")
        print(f"  Time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Duration: {duration/60:.1f} minutes")
        print(f"  Events: {len(events)} total")
        
        # Show event type breakdown
        main_types = ['caption', 'mood', 'reflection', 'drawing_decision', 'motif_score', 'motif_extraction', 'motif_analysis', 'tinyllama_score', 'error']
        type_summary = []
        for t in main_types:
            if t in event_types:
                if t in ['tinyllama_score', 'motif_score']:
                    type_summary.append(f"{event_types[t]} motif scores")
                elif t in ['motif_extraction', 'motif_analysis']:
                    type_summary.append(f"{event_types[t]} motif events")
                elif t == 'error':
                    type_summary.append(f"{event_types[t]} errors")
                else:
                    type_summary.append(f"{event_types[t]} {t}s")
        if type_summary:
            print(f"  Types: {', '.join(type_summary)}")
            
        # Show sample caption
        if captions:
            print(f"  Sample: \"{captions[0]}\"")
            
        print()
    
    def view_run(self, run_id: str, event_filter: Optional[str] = None) -> None:
        """View detailed events from a specific run."""
        if run_id not in self.logs:
            print(f"Run {run_id} not found!")
            return
            
        events = self.logs[run_id]
        print(f"\n{'='*80}")
        print(f"Run {run_id} - {len(events)} events")
        print(f"{'='*80}\n")
        
        for i, event in enumerate(events):
            if event_filter and event_filter not in str(event):
                continue
                
            self._print_event(i, event)
            
    def _print_event(self, index: int, event: Dict) -> None:
        """Print a single event."""
        timestamp = event.get('timestamp', 0)
        event_type = event.get('type', 'unknown')
        data = event.get('data', {})
        
        time_str = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
        
        # Color coding for different event types
        if event_type == 'caption':
            print(f"[{time_str}] [CAP] CAPTION")
            # Caption can be directly in event or in data field
            caption = event.get('caption', '') or data.get('caption', '')
            if caption:
                print(f"  {caption}")
        elif event_type == 'mood':
            print(f"[{time_str}] [MOOD] MOOD")
            if 'mood' in data:
                print(f"  Mood: {data['mood']:.2f}")
            if 'emotion' in data:
                print(f"  Emotion: {data['emotion']}")
        elif event_type == 'reflection':
            print(f"[{time_str}] [REF] REFLECTION")
            if 'reflection' in data:
                print(f"  {data['reflection']}")
        elif event_type == 'drawing_decision':
            print(f"[{time_str}] [ART] DRAWING")
            if 'will_draw' in data:
                print(f"  Will draw: {data['will_draw']}")
            if 'reason' in data:
                print(f"  Reason: {data['reason']}")
        elif event_type == 'snapshot':
            print(f"[{time_str}] [IMG] SNAPSHOT")
            if 'image_path' in data:
                print(f"  Image: {data['image_path']}")
        elif event_type == 'motif_score':
            print(f"[{time_str}] [MOT] MOTIF SCORE")
            if 'motif' in data:
                print(f"  Motif: {data['motif']}")
            if 'score' in data:
                print(f"  Score: {data['score']}")
            if 'method' in data:
                print(f"  Method: {data['method']}")
            if 'elapsed_ms' in data:
                print(f"  Time: {data['elapsed_ms']}ms")
            if 'cache_stats' in data:
                cache = data['cache_stats']
                print(f"  Cache: {cache.get('hit_rate', 0):.1%} hit rate")
        elif event_type == 'motif_extraction':
            print(f"[{time_str}] [EXT] MOTIF EXTRACTION")
            if 'motifs_found' in data:
                print(f"  Found: {len(data['motifs_found'])} motifs")
                if data['motifs_found']:
                    print(f"  Top: {', '.join(data['motifs_found'][:3])}")
            if 'extraction_time_ms' in data:
                print(f"  Time: {data['extraction_time_ms']}ms")
        elif event_type == 'motif_analysis':
            print(f"[{time_str}] [ANA] MOTIF ANALYSIS")
            if 'analyzed_motifs' in data:
                print(f"  Analyzed: {len(data['analyzed_motifs'])}")
            if 'high_significance' in data:
                high_sig = data['high_significance']
                if high_sig:
                    print(f"  Significant: {', '.join(high_sig[:3])}")
        elif event_type == 'tinyllama_score':
            print(f"[{time_str}] [TLL] TINYLLAMA")
            if 'motif' in data:
                print(f"  Motif: {data['motif']}")
            if 'score' in data:
                print(f"  Score: {data['score']}")
            if 'context' in data:
                context = data['context'][:60] + "..." if len(str(data['context'])) > 60 else data['context']
                print(f"  Context: {context}")
        elif event_type == 'error':
            print(f"[{time_str}] [ERR] ERROR")
            if 'message' in data:
                print(f"  {data['message']}")
            if 'component' in data:
                print(f"  Component: {data['component']}")
        else:
            print(f"[{time_str}] [{event_type[:3].upper()}] {event_type.upper()}")
            if data:
                for key, value in data.items():
                    if isinstance(value, str) and len(value) > 100:
                        value = value[:100] + "..."
                    print(f"  {key}: {value}")
        print()
    
    def search_events(self, query: str) -> None:
        """Search for events containing specific text."""
        print(f"\n{'='*80}")
        print(f"Searching for: '{query}'")
        print(f"{'='*80}\n")
        
        matches = []
        for run_id, events in self.logs.items():
            for event in events:
                if query.lower() in json.dumps(event).lower():
                    matches.append((run_id, event))
                    
        print(f"Found {len(matches)} matches\n")
        
        for run_id, event in matches[:50]:  # Limit to first 50
            print(f"Run {run_id}:")
            self._print_event(0, event)
            
    def show_statistics(self) -> None:
        """Show overall statistics across all runs."""
        print(f"\n{'='*80}")
        print("OVERALL STATISTICS")
        print(f"{'='*80}\n")
        
        total_events = 0
        total_captions = 0
        total_reflections = 0
        total_drawings = 0
        total_tinyllama_scores = 0
        total_motif_scores = 0
        total_errors = 0
        all_emotions = []
        all_moods = []
        
        for events in self.logs.values():
            total_events += len(events)
            for event in events:
                event_type = event.get('type', '')
                data = event.get('data', {})
                
                if event_type == 'caption':
                    total_captions += 1
                elif event_type == 'reflection':
                    total_reflections += 1
                elif event_type == 'drawing_decision' and data.get('will_draw'):
                    total_drawings += 1
                elif event_type == 'tinyllama_score':
                    total_tinyllama_scores += 1
                elif event_type in ['motif_score', 'motif_extraction', 'motif_analysis']:
                    total_motif_scores += 1
                elif event_type == 'error':
                    total_errors += 1
                elif event_type == 'mood':
                    if 'emotion' in data:
                        all_emotions.append(data['emotion'])
                    if 'mood' in data:
                        all_moods.append(data['mood'])
        
        print(f"Total runs: {len(self.logs)}")
        print(f"Total events: {total_events:,}")
        print(f"Total captions: {total_captions:,}")
        print(f"Total reflections: {total_reflections:,}")
        print(f"Total drawings initiated: {total_drawings}")
        print(f"Total TinyLlama scores: {total_tinyllama_scores}")
        print(f"Total motif events: {total_motif_scores}")
        print(f"Total errors logged: {total_errors}")
        
        if all_moods:
            avg_mood = sum(all_moods) / len(all_moods)
            print(f"\nAverage mood: {avg_mood:.2f}")
            
        if all_emotions:
            from collections import Counter
            emotion_counts = Counter(all_emotions)
            print(f"\nTop emotions:")
            for emotion, count in emotion_counts.most_common(5):
                print(f"  {emotion}: {count} times")


def main():
    parser = argparse.ArgumentParser(description='View LOVE_YOURSELF event logs')
    parser.add_argument('--recent', type=int, help='Show N most recent runs')
    parser.add_argument('--run', type=str, help='View specific run by ID')
    parser.add_argument('--search', type=str, help='Search for events containing text')
    parser.add_argument('--stats', action='store_true', help='Show overall statistics')
    parser.add_argument('--filter', type=str, help='Filter events when viewing a run')
    
    args = parser.parse_args()
    
    viewer = LogViewer()
    
    if args.stats:
        viewer.show_statistics()
    elif args.search:
        viewer.search_events(args.search)
    elif args.run:
        viewer.view_run(args.run, args.filter)
    elif args.recent:
        viewer.list_runs(args.recent)
    else:
        # Default: show recent runs
        viewer.list_runs(20)
        print("\nUsage examples:")
        print("  python debug/log_viewer.py --recent 10")
        print("  python debug/log_viewer.py --run abc123")
        print("  python debug/log_viewer.py --search 'man sitting'")
        print("  python debug/log_viewer.py --stats")


if __name__ == "__main__":
    main()