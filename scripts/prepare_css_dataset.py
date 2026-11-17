"""Prepare either CSS environment from an existing RP 70/15/15 split, without starting SUMO or training."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict, deque
from pathlib import Path

sys.dont_write_bytecode = True
SPLITS = ('training', 'validation', 'testing')
DEFAULT_SOURCE = Path(r'C:\Users\Master\Documents\Codex\2026-08-22\new-chat\RP_refactor\new_env')
DEFAULT_STATIONS = ['1419061109', '446290091', '5984003368']


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def power_seed(base_seed, flow, penetration, background_seed, ev_id):
    key = '|'.join(map(str, (base_seed, flow, penetration, background_seed, ev_id)))
    return int.from_bytes(hashlib.sha256(key.encode('utf-8')).digest()[:16], 'big')


def sample_soc(rng, mean=0.30, sd=0.05, low=0.20, high=0.40):
    if not all(math.isfinite(x) for x in (mean, sd, low, high)) or not low < mean < high or sd <= 0:
        raise ValueError('SOC requires finite low < mean < high and sd > 0')
    while True:
        value = rng.gauss(mean, sd)
        if low <= value <= high:
            return round(value, 6)


def vehicle_records(path):
    root = ET.parse(path).getroot()
    records = []
    for vehicle in root.findall('vehicle'):
        vid = vehicle.get('id', '')
        if not vid.startswith('EV'):
            raise ValueError(f'Unexpected non-EV vehicle in EV routes: {path}: {vid}')
        route = vehicle.find('route')
        edges = route.get('edges', '').split() if route is not None else []
        if not edges:
            raise ValueError(f'Missing embedded EV route: {path}: {vid}')
        records.append((vid, edges[0], edges))
    if not records or len({x[0] for x in records}) != len(records):
        raise ValueError(f'Empty/duplicate EV IDs: {path}')
    return records


def connection_reachability(net, stations):
    edges = {e.getID(): e for e in net.getEdges() if not e.getID().startswith(':') and e.allows('passenger')}
    reverse = defaultdict(set)
    for eid, edge in edges.items():
        for following, connections in edge.getOutgoing().items():
            fid = following.getID()
            if fid in edges and any(c.allows('passenger') for c in connections):
                reverse[fid].add(eid)
    reachable = {}
    for station in stations:
        node = net.getNode(station)
        seen = {e.getID() for e in node.getIncoming() if e.getID() in edges}
        if not seen:
            raise ValueError(f'No passenger-accessible incoming edge: {station}')
        queue = deque(seen)
        while queue:
            for previous in reverse[queue.popleft()]:
                if previous not in seen:
                    seen.add(previous)
                    queue.append(previous)
        reachable[station] = seen
    return reachable


def summarize(values):
    quantiles = statistics.quantiles(values, n=100, method='inclusive')
    return {'count': len(values), 'mean': statistics.fmean(values), 'sample_sd': statistics.stdev(values),
            'minimum': min(values), 'p05': quantiles[4], 'median': statistics.median(values),
            'p95': quantiles[94], 'maximum': max(values)}


def safe_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha(source) != sha(target):
            raise FileExistsError(f'Refusing to replace different file: {target}')
    else:
        shutil.copy2(source, target)
    if sha(source) != sha(target):
        raise RuntimeError(f'Copy hash mismatch: {target}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stations', nargs='+', required=True)
    parser.add_argument('--power-seed', type=int, default=20261107)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    stations = sorted(set(args.stations))
    if source == output or output in source.parents or source in output.parents:
        raise ValueError('Source and output directories must be separate trees')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use an empty output directory; existing datasets are never overwritten')
    from sumolib.net import readNet
    scenarios = read_csv(source / 'scenario_manifest.csv')
    if len(scenarios) != 600:
        raise ValueError(f'Expected 600 existing RP scenarios, got {len(scenarios)}')
    flows = sorted({int(r['flow_veh_h']) for r in scenarios})
    if len(flows) != 3:
        raise ValueError('Expected three demand levels')
    seed_splits = {}
    seen = set()
    for row in scenarios:
        split = row['split']
        seed = int(row['background_seed'])
        key = (int(row['flow_veh_h']), seed, row['penetration'])
        if split not in SPLITS or key in seen or row['penetration'] not in ('p01', 'p02'):
            raise ValueError(f'Invalid/duplicate source scenario: {row}')
        seen.add(key)
        if seed in seed_splits and seed_splits[seed] != split:
            raise ValueError(f'Cross-split generator seed: {seed}')
        seed_splits[seed] = split
        for field in ('general_route_file', 'ev_route_file'):
            path = (source / row[field]).resolve()
            if path.parent != source / f'{split}_data' or not path.is_file():
                raise ValueError(f'Invalid source route path: {row[field]}')
    for split in SPLITS:
        group = Counter((int(r['flow_veh_h']), r['penetration']) for r in scenarios if r['split'] == split)
        expected = 70 if split == 'training' else 15
        if group != {(f, p): expected for f in flows for p in ('p01','p02')}:
            raise ValueError(f'Invalid split sizes: {split}: {group}')
    net = readNet(str(source / 'env_static_tls.net.xml'), withInternal=True)
    reachable = connection_reachability(net, stations)
    prepared, assignments = [], []
    reference_matches = 0
    for row in scenarios:
        split, flow, penetration, background = row['split'], int(row['flow_veh_h']), row['penetration'], int(row['background_seed'])
        if split not in SPLITS:
            raise ValueError(f'Unknown split: {split}')
        records = vehicle_records(source / row['ev_route_file'])
        if len(records) != int(row['ev_vehicle_count']):
            raise ValueError('EV count does not match source manifest')
        goal_rng, goals, powers = random.Random(background), [], []
        for order, (vid, first, _) in enumerate(records):
            candidates = [station for station in stations if first in reachable[station]]
            if not candidates:
                raise ValueError(f'No reachable station: {row["ev_route_file"]}: {vid}: {first}')
            goal = goal_rng.choice(candidates)
            seed = power_seed(args.power_seed, flow, penetration, background, vid)
            soc = sample_soc(random.Random(seed))
            goals.append(goal)
            powers.append(soc)
            assignments.append({'split': split, 'flow_veh_h': flow, 'background_seed': background,
                                'penetration': penetration, 'ev_order': order, 'ev_id': vid, 'first_edge': first,
                                'goal_node': goal, 'initial_soc': soc, 'reachable_station_count': len(candidates),
                                'reachable_stations': '|'.join(candidates), 'power_rng_seed': str(seed)})
        suffix = f'ev_{flow}_{penetration}_{background}.json'
        goal_file, power_file = f'{split}_data_goal/goal_{suffix}', f'{split}_data_power/power_{suffix}'
        reference = source / goal_file
        if reference.exists() and json.loads(reference.read_text(encoding='utf-8')) == goals:
            reference_matches += 1
        prepared.append((row, records, goals, powers, goal_file, power_file))
    output.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        for ending in ('data', 'data_goal', 'data_power'):
            (output / f'{split}_{ending}').mkdir()
    copy_paths = {'env_static_tls.net.xml', 'osm.sumocfg', 'scenario_manifest.csv', 'background_split_manifest.csv'}
    for row, *_ in prepared:
        copy_paths.update((row['general_route_file'], row['ev_route_file']))
    for name in sorted(copy_paths):
        safe_copy(source / name, output / name)
    data_rows = []
    for row, records, goals, powers, goal_file, power_file in prepared:
        write_json(output / goal_file, goals)
        write_json(output / power_file, powers)
        data_rows.append(dict(row) | {'goal_file': goal_file, 'power_file': power_file,
                                     'ev_route_sha256': sha(output / row['ev_route_file']),
                                     'goal_sha256': sha(output / goal_file), 'power_sha256': sha(output / power_file)})
    write_json(output / 'cs_locations.json', stations)
    write_csv(output / 'data_manifest.csv', data_rows)
    write_csv(output / 'ev_assignments.csv', assignments)
    split_protocol = json.loads((source / 'split_protocol.json').read_text(encoding='utf-8'))
    split_protocol.update({'source': str(source), 'destination': str(output), 'flows_veh_h': flows,
                           'operation': 'copy existing RP partition; retain all source files',
                           'included': '300 background routes, 600 EV routes, network/config; generated 600 goal and 600 SOC files',
                           'excluded': 'training, simulator execution, runtime code changes, model checkpoints'})
    write_json(output / 'split_protocol.json', split_protocol)
    protocol = {'source_rp_dataset': str(source), 'network_sha256': sha(output / 'env_static_tls.net.xml'),
                'stations': stations, 'stations_are': 'synthetic charging-station locations at user-selected junctions; not verified real facilities',
                'goal_policy': 'uniform random among passenger/turn-reachable stations from actual first route edge',
                'goal_seed_policy': 'random.Random(background_seed), reset for each scenario; sorted candidate station IDs; EV XML order',
                'goal_lists_matching_existing_RP': reference_matches,
                'soc_distribution': {'family': 'truncated normal by rejection sampling, then round to 6 decimal places',
                                     'underlying_mean': 0.30, 'underlying_sd': 0.05, 'lower': 0.20, 'upper': 0.40,
                                     'units': 'battery state-of-charge fraction; 0.20 = 20%',
                                     'note': 'Underlying SD is before truncation; realized SD is smaller. No boundary clipping.'},
                'soc_base_seed': args.power_seed,
                'soc_seed_policy': 'first 16 SHA256 bytes as big-endian integer of base_seed|flow|penetration|background_seed|actual_ev_id',
                'soc_rng': 'Python random.Random(seed).gauss(mean, sd), reject outside limits',
                'list_alignment': 'actual XML EV ID order, never assume contiguous EV numbers; explicit mapping in ev_assignments.csv',
                'runtime_status': 'Select this dataset with the manifest-based CSS environment loader.',
                'goal_usage': 'Current CSS agent chooses station at departure and does not load goal files. Saved random goals are reference/baseline assignments.',
                'simulation_performed': False, 'training_performed': False}
    write_json(output / 'generation_protocol.json', protocol)
    values = [float(a['initial_soc']) for a in assignments]
    stats = {'overall': summarize(values), 'by_split': {s: summarize([float(a['initial_soc']) for a in assignments if a['split'] == s]) for s in SPLITS}}
    write_json(output / 'power_statistics.json', stats)
    distribution = []
    for split in SPLITS:
        for flow in flows:
            group = [a for a in assignments if a['split'] == split and a['flow_veh_h'] == flow]
            counts = Counter(a['goal_node'] for a in group)
            for station in stations:
                distribution.append({'split': split, 'flow_veh_h': flow, 'station_node': station,
                                     'ev_count': counts[station], 'group_ev_count': len(group),
                                     'fraction': counts[station] / len(group)})
    write_csv(output / 'goal_distribution.csv', distribution)
    counts = {s: {'background_routes': len({r['general_route_file'] for r in scenarios if r['split'] == s}),
                  'ev_routes': sum(r['split'] == s for r in scenarios),
                  'ev_assignments': sum(a['split'] == s for a in assignments)} for s in SPLITS}
    readme = f"""# CSS 70/15/15 dataset

Demands: {flows}. Each demand has 100 background seeds, split 70/15/15.
Each background and its p01/p02 EV variants stay in the same partition.
Identical numeric generator seeds across demands also stay in the same partition.
The partition and routes are copied exactly from RP. This is the full dataset,
without post-testing remove3/remove4 exclusions.

Counts: {json.dumps(counts)}

Stations: {stations}, synthetic locations on the original network.
Reference goals are sampled uniformly from passenger/turn-reachable stations.
The CSS agent still selects its own station; reference goal lists do not override it.
Initial SOC is rejection-sampled N(0.30, 0.05^2), restricted to [0.20, 0.40],
rounded to six decimals. Base seed {args.power_seed}; SHA256-derived per-EV RNG.
Every JSON list follows actual EV XML order; ev_assignments.csv records the mapping.
See generation_protocol.json for exact reproduction and data_manifest.csv for hashes.

Audit: python verify_dataset.py --root .
Regenerate into an empty directory:
python prepare_dataset.py --source <RP dataset> --output <empty directory> --stations {' '.join(stations)}

Source files are preserved. No training has been started by dataset preparation.
"""
    (output / 'README_zh-TW.md').write_text(readme, encoding='utf-8')
    safe_copy(Path(__file__), output / 'prepare_dataset.py')
    verifier = Path(__file__).resolve().parent / 'verify_css_dataset.py'
    if not verifier.exists():
        verifier = Path(__file__).resolve().parent / 'verify_dataset.py'
    safe_copy(verifier, output / 'verify_dataset.py')
    manifest = []
    for path in sorted(output.rglob('*')):
        if path.is_file():
            manifest.append({'file': path.relative_to(output).as_posix(), 'bytes': path.stat().st_size, 'sha256': sha(path)})
    write_csv(output / 'dataset_file_manifest.csv', manifest)
    counts = {s: {'background_routes': len({r['general_route_file'] for r in scenarios if r['split'] == s}),
                  'ev_routes': sum(r['split'] == s for r in scenarios),
                  'ev_assignments': sum(a['split'] == s for a in assignments)} for s in SPLITS}
    print(json.dumps({'status': 'prepared', 'output': str(output), 'counts': counts, 'goals_matching_rp': reference_matches,
                      'power_statistics': stats['overall'], 'simulation_performed': False}, indent=2), flush=True)


if __name__ == '__main__':
    main()
