"""Prepare CSS new_env from the existing RP split, without starting SUMO or training."""
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
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stations', nargs='+', default=DEFAULT_STATIONS)
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
    split_protocol.update({'source': str(source), 'destination': str(output),
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
                'runtime_status': 'Data prepared only. Current CSS defaults still use env and old seed ranges/stations; new_env runtime integration is separate.',
                'goal_usage': 'Current CSS agent chooses station at departure and does not load goal files. Saved random goals are reference/baseline assignments.',
                'simulation_performed': False, 'training_performed': False}
    write_json(output / 'generation_protocol.json', protocol)
    values = [float(a['initial_soc']) for a in assignments]
    stats = {'overall': summarize(values), 'by_split': {s: summarize([float(a['initial_soc']) for a in assignments if a['split'] == s]) for s in SPLITS}}
    write_json(output / 'power_statistics.json', stats)
    distribution = []
    for split in SPLITS:
        for flow in (2650, 2800, 2950):
            group = [a for a in assignments if a['split'] == split and a['flow_veh_h'] == flow]
            counts = Counter(a['goal_node'] for a in group)
            for station in stations:
                distribution.append({'split': split, 'flow_veh_h': flow, 'station_node': station,
                                     'ev_count': counts[station], 'group_ev_count': len(group),
                                     'fraction': counts[station] / len(group)})
    write_csv(output / 'goal_distribution.csv', distribution)
    readme = '''# CSS new_env 資料集

資料沿用 RP 的既有 70% / 15% / 15% 切分，沒有重新抽樣。

| split | 背景車流檔 | EV 車流檔／goal 檔／power 檔（各） | EV 筆數 |
|---|---:|---:|---:|
| training | 210 | 420 | 8890 |
| validation | 45 | 90 | 1905 |
| testing | 45 | 90 | 1905 |
| 合計 | 300 | 600 | 12700 |

需求為 2650、2800、2950；每種需求 100 個 seed，每個 seed 有 p01、p02 兩種 EV 車流。
同一背景 seed、兩種 EV 比例與跨需求共用 seed 的 partition 都與 RP 完全相同。
原始 RP/new-env 檔案保留；本資料集採複製而非刪除來源。

充電站採使用者指定的 junction：1419061109、446290091、5984003368。
這是模擬充電站位置，並非宣稱當地有真實充電站。
隨機 goal 從 passenger 車輛依合法轉向可達的站點中等機率選擇；固定背景 seed，
保持與 RP 的 goal 生成政策一致。這份資料不是強制每站分到完全相同數量。

power 是初始 SOC 比例，0.2 = 20%。以平均 0.30、原始標準差 0.05 的常態分布抽樣，
超出 [0.20, 0.40] 重新抽取，屬截斷常態分布；不是把超界數值裁切成端點。
截斷後實際 SD 小於 0.05，實測分布見 power_statistics.json。
固定 base seed 20261107；每個場景及真實 EV ID 的 SOC seed 獨立且可重現。
所有演算法讀取同一組檔案即可共用相同起始 SOC。

goal_*.json 和 power_*.json 均為 JSON list，順序逐一對齊 EV .rou.xml 的實際車輛順序；
不得用連續 EV_0、EV_1 假設來代替 XML 中的 ID。ev_assignments.csv 保存逐車對應、
起點 edge、goal、SOC、可達站點及 SOC seed。data_manifest.csv 保存每個場景各檔案位置與雜湊。

目前只完成資料準備，沒有修改 CSS 訓練流程，也沒有啟動模擬或訓練。
現行 CSS 主流程仍使用 env、舊需求／seed 範圍及舊站點，尚未提供 new_env 選擇。
現行 CSS 由 agent 在出發時選擇充電站，不讀 goal 檔；隨機 goal 檔供固定隨機基準或後續串接。
若把固定 goal 強制用在學習方法上，會覆蓋其充電站選擇動作，因此此處不改動選站流程。
網路與 osm.sumocfg 沿用 RP；config 未指定 route-files，需要由後續 loader 明確指定。
使用新路網訓練時，也需使用相容的新路網路徑規劃模型，不能直接沿用舊 Brooklyn 的 checkpoint。

重建（使用空的新目錄，避免覆蓋）：

    python prepare_dataset.py --source <RP_refactor/new_env> --output <empty_new_directory>

完整生成規則見 generation_protocol.json；完整查核結果見 verification.json。

靜態驗證（需能讀取原 RP 資料來源）：

    python verify_dataset.py --root .

生成器測試：

    python -B test_generation.py
'''
    (output / 'README_zh-TW.md').write_text(readme, encoding='utf-8')
    safe_copy(Path(__file__), output / 'prepare_dataset.py')
    for helper in ('verify_dataset.py', 'test_generation.py'):
        helper_path = Path(__file__).resolve().parent / helper
        if helper_path.is_file():
            safe_copy(helper_path, output / helper)
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
