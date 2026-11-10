#!/usr/bin/env python3
"""Read-only ACTION7 mechanism report for saved offline Flash artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def diagnose(root: Path) -> dict:
    benchmark = json.loads((root / 'benchmark.json').read_text())
    games = []
    for run in benchmark['game_runs']:
        game_id = run['game_id']
        # Do not treat response metadata and repeated prompts as additional calls.
        paths = sorted((root / 'transcripts').glob(f'{game_id}_p*.txt'))
        sections = []
        for path in paths:
            parts = re.split(r'\n\[([^\]\n]+)\]\n', path.read_text())
            sections.extend(zip(parts[1::2], parts[2::2]))
        mentions = sum(tag == 'TOOL CALL: python' and bool(re.search(r'\bACTION7\b', text))
                       for tag, text in sections)
        executions = sum(item.get('action', {}).get('id') == 'ACTION7'
                         for item in run['history'])
        games.append({
            'game_id': game_id,
            'transcripts_present': bool(paths),
            'tool_snippets_mentioning_action7': mentions,
            'executed_action7': executions,
        })
    return {
        'evidence': 'offline mechanism diagnostic, not a score improvement',
        'note': 'Snippet mentions include potential branches/loops; they are not exact attempted-action counts.',
        'game_count': len(games),
        'transcript_game_count': sum(g['transcripts_present'] for g in games),
        'tool_snippets_mentioning_action7': sum(g['tool_snippets_mentioning_action7'] for g in games),
        'executed_action7': sum(g['executed_action7'] for g in games),
        'games': games,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir', type=Path)
    print(json.dumps(diagnose(parser.parse_args().output_dir), indent=2))
