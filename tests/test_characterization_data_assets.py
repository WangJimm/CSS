from __future__ import annotations

import json
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_ROOT = PROJECT_ROOT / "env"


def ev_count(path: Path) -> int:
    return sum(
        (vehicle.get("id") or "").startswith("EV")
        for vehicle in ET.parse(path).getroot().findall("vehicle")
    )


class DataAssetCharacterizationTests(unittest.TestCase):
    def test_partition_contains_exactly_the_manifest_routes(self):
        import csv
        with (ENV_ROOT / 'scenario_manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        for split in ('training', 'validation', 'testing'):
            expected = {row['ev_route_file'] for row in rows if row['split'] == split}
            actual = {p.relative_to(ENV_ROOT).as_posix() for p in (ENV_ROOT / f'{split}_data').glob('ev_*.rou.xml')}
            self.assertEqual(actual, expected)
            self.assertEqual(len(actual), 420 if split == 'training' else 90)

    def test_all_power_files_match_route_ev_cardinality(self):
        mismatches = []
        for split in ("training", "testing", "validation"):
            route_dir = ENV_ROOT / f"{split}_data"
            power_dir = ENV_ROOT / f"{split}_data_power"
            for route_path in sorted(route_dir.glob("ev_*.rou.xml")):
                scenario = route_path.name[len("ev_") : -len(".rou.xml")]
                power_path = power_dir / f"power_ev_{scenario}.json"
                powers = json.loads(power_path.read_text(encoding="utf-8"))
                if ev_count(route_path) != len(powers):
                    mismatches.append((split, route_path.name, ev_count(route_path), len(powers)))

        self.assertEqual(mismatches, [])

    def test_default_sumo_configuration_route_exists(self):
        config_path = ENV_ROOT / "osm.sumocfg"
        root = ET.parse(config_path).getroot()
        self.assertTrue((ENV_ROOT / root.find('./input/net-file').get('value')).is_file())
        # Runtime explicitly supplies the selected background and EV routes.
        self.assertIsNone(root.find('./input/route-files'))


if __name__ == "__main__":
    unittest.main()
