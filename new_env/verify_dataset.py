"""Audit the complete CSS new_env dataset without launching SUMO."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import random
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

sys.dont_write_bytecode = True


def load_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()
    read_json = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
    protocol = read_json(root / 'generation_protocol.json')
    source = Path(protocol['source_rp_dataset'])
    rows, original = load_csv(root / 'scenario_manifest.csv'), load_csv(source / 'scenario_manifest.csv')
    assert rows == original and len(rows) == 600, 'Partition/source scenario mismatch'
    stations = read_json(root / 'cs_locations.json')
    assert stations == protocol['stations'] and stations == sorted(set(stations))
    assert digest(root / 'env_static_tls.net.xml') == digest(source / 'env_static_tls.net.xml') == protocol['network_sha256']
    assert digest(root / 'osm.sumocfg') == digest(source / 'osm.sumocfg')
    cfg = ET.parse(root / 'osm.sumocfg').getroot()
    assert (root / cfg.find('./input/net-file').get('value')).is_file()
    from sumolib.net import readNet
    net = readNet(str(root / 'env_static_tls.net.xml'), withInternal=True)
    external = {e.getID(): e for e in net.getEdges() if not e.getID().startswith(':') and e.allows('passenger')}
    incoming = {station: [e for e in net.getNode(station).getIncoming() if e.getID() in external] for station in stations}
    assert all(incoming.values())
    @lru_cache(maxsize=None)
    def shortest_reachable(first, station):
        return any(net.getShortestPath(net.getEdge(first), end, vClass='passenger')[0] is not None for end in incoming[station])
    @lru_cache(maxsize=None)
    def legal_pair(first, second):
        return second in external and any(following.getID() == second and any(c.allows('passenger') for c in links)
                                          for following, links in external[first].getOutgoing().items())
    @lru_cache(maxsize=None)
    def check_route(path):
        xml_root = ET.parse(path).getroot()
        vehicles = xml_root.findall('vehicle')
        ids = [v.get('id') for v in vehicles]
        assert len(set(ids)) == len(ids), f'Duplicate route vehicle IDs: {path}'
        for vehicle in vehicles:
            route = vehicle.find('route')
            assert route is not None
            edges = route.get('edges', '').split()
            assert edges and all(e in external for e in edges), f'Unknown/restricted route edge: {path}'
            assert all(legal_pair(a, b) for a, b in zip(edges, edges[1:])), f'Illegal passenger turn in {path}'
        return vehicles
    spec = importlib.util.spec_from_file_location('css_dataset_generator', root / 'prepare_dataset.py')
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    assignments = load_csv(root / 'ev_assignments.csv')
    keyed = {(a['split'], a['flow_veh_h'], a['penetration'], a['background_seed'], a['ev_id']): a for a in assignments}
    assert len(keyed) == len(assignments) == 12700
    manifest_rows = load_csv(root / 'data_manifest.csv')
    assert len(manifest_rows) == 600
    data_manifest = {a['ev_route_file']: a for a in manifest_rows}
    route_paths, goals_paths, powers_paths = set(), set(), set()
    seeds_by_split, hashes_by_split = defaultdict(set), defaultdict(set)
    counts, station_counts, candidate_counts = {}, Counter(), Counter()
    powers_all = []
    goal_reference_matches = 0
    for number, row in enumerate(rows, 1):
        split, flow, penetration, seed = row['split'], int(row['flow_veh_h']), row['penetration'], int(row['background_seed'])
        seeds_by_split[split].add(seed)
        for route_kind in ('general_route_file', 'ev_route_file'):
            path = root / row[route_kind]
            value = digest(path)
            assert value == digest(source / row[route_kind]), f'Source copy mismatch: {path}'
            hashes_by_split[split].add(value)
            route_paths.add(path.relative_to(root).as_posix())
        backgrounds = check_route(str(root / row['general_route_file']))
        vehicles = check_route(str(root / row['ev_route_file']))
        assert len(backgrounds) == int(row['background_vehicle_count'])
        assert len(vehicles) == int(row['ev_vehicle_count'])
        assert not ({v.get('id') for v in backgrounds} & {v.get('id') for v in vehicles})
        suffix = f'ev_{flow}_{penetration}_{seed}.json'
        goal_name, power_name = f'{split}_data_goal/goal_{suffix}', f'{split}_data_power/power_{suffix}'
        goals, powers = read_json(root / goal_name), read_json(root / power_name)
        assert isinstance(goals, list) and isinstance(powers, list) and len(goals) == len(powers) == len(vehicles)
        goals_paths.add(goal_name); powers_paths.add(power_name)
        rng = random.Random(seed)
        for i, (vehicle, goal, power) in enumerate(zip(vehicles, goals, powers)):
            vid = vehicle.get('id')
            first = vehicle.find('route').get('edges').split()[0]
            valid = [station for station in stations if shortest_reachable(first, station)]
            assert goal in valid and goal == rng.choice(valid), f'Goal reachability/randomness mismatch: {vid}'
            assert isinstance(power, (int, float)) and not isinstance(power, bool) and math.isfinite(power) and .2 <= power <= .4
            power_seed = generator.power_seed(protocol['soc_base_seed'], flow, penetration, seed, vid)
            assert power == generator.sample_soc(random.Random(power_seed))
            entry = keyed[(split, str(flow), penetration, str(seed), vid)]
            assert int(entry['ev_order']) == i and entry['first_edge'] == first and entry['goal_node'] == goal
            assert float(entry['initial_soc']) == power and int(entry['power_rng_seed']) == power_seed
            assert entry['reachable_stations'].split('|') == valid and int(entry['reachable_station_count']) == len(valid)
            powers_all.append(power); station_counts[goal] += 1; candidate_counts[len(valid)] += 1
        reference = source / goal_name
        if reference.is_file() and read_json(reference) == goals:
            goal_reference_matches += 1
        detail = data_manifest[row['ev_route_file']]
        assert all(detail[k] == value for k, value in row.items())
        assert detail['goal_file'] == goal_name and detail['power_file'] == power_name
        for path_key, hash_key in [('ev_route_file', 'ev_route_sha256'), ('goal_file', 'goal_sha256'), ('power_file', 'power_sha256')]:
            assert digest(root / detail[path_key]) == detail[hash_key]
        if number % 100 == 0:
            print(f'AUDIT {number}/600 scenarios', flush=True)
    for split in ('training', 'validation', 'testing'):
        expected = 420 if split == 'training' else 90
        selected = [r for r in rows if r['split'] == split]
        assert len(selected) == expected
        groups = Counter((int(r['flow_veh_h']), r['penetration']) for r in selected)
        assert groups == {(f, p): (70 if split == 'training' else 15) for f in (2650, 2800, 2950) for p in ('p01', 'p02')}
        counts[split] = {'background_routes': len({r['general_route_file'] for r in selected}), 'ev_routes': len(selected),
                         'goal_files': sum(name.startswith(split + '_') for name in goals_paths),
                         'power_files': sum(name.startswith(split + '_') for name in powers_paths),
                         'ev_assignments': sum(int(r['ev_vehicle_count']) for r in selected)}
    for a, b in [('training','validation'), ('training','testing'), ('validation','testing')]:
        assert not seeds_by_split[a] & seeds_by_split[b]
        assert not hashes_by_split[a] & hashes_by_split[b]
    assert route_paths == {p.relative_to(root).as_posix() for split in counts for p in (root / f'{split}_data').glob('*.rou.xml')}
    assert goals_paths == {p.relative_to(root).as_posix() for split in counts for p in (root / f'{split}_data_goal').glob('*.json')}
    assert powers_paths == {p.relative_to(root).as_posix() for split in counts for p in (root / f'{split}_data_power').glob('*.json')}
    hashes = load_csv(root / 'dataset_file_manifest.csv')
    assert len({r['file'] for r in hashes}) == len(hashes)
    expected_files = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name not in ('dataset_file_manifest.csv', 'verification.json')}
    assert {r['file'] for r in hashes} == expected_files
    for item in hashes:
        path = root / item['file']
        assert path.stat().st_size == int(item['bytes']) and digest(path) == item['sha256']
    stats = generator.summarize(powers_all)
    assert stats == read_json(root / 'power_statistics.json')['overall']
    assert .297 < stats['mean'] < .303 and .041 < stats['sample_sd'] < .047
    report = {'status': 'passed', 'dataset_root': str(root), 'scenario_count': len(rows), 'counts': counts,
              'ev_assignments': len(assignments), 'source_route_sha256_verified': len(route_paths),
              'route_xml_and_passenger_turns_verified': check_route.cache_info().currsize,
              'unique_legal_route_turn_pairs': legal_pair.cache_info().currsize,
              'all_goals_reachable': True, 'independent_reachability_check': 'sumolib passenger shortest paths; generator used reverse connection BFS',
              'goals_matching_existing_RP': goal_reference_matches, 'station_assignment_counts': dict(station_counts),
              'reachable_station_count_histogram': dict(candidate_counts),
              'all_ev_ids_and_list_order_verified': True, 'all_soc_rng_values_reproduced': True,
              'power_statistics': stats, 'no_cross_split_seed_or_hash_overlap': True,
              'file_checksums_verified': len(hashes), 'source_files_retained': True,
              'runtime_code_changed': False, 'simulation_performed': False, 'training_performed': False}
    (root / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
