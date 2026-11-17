# CSS 70/15/15 dataset

Demands: [1500, 1800, 2100]. Each demand has 100 background seeds, split 70/15/15.
Each background and its p01/p02 EV variants stay in the same partition.
Identical numeric generator seeds across demands also stay in the same partition.
The partition and routes are copied exactly from RP. This is the full dataset,
without post-testing remove3/remove4 exclusions.

Counts: {"training": {"background_routes": 210, "ev_routes": 420, "ev_assignments": 5740}, "validation": {"background_routes": 45, "ev_routes": 90, "ev_assignments": 1230}, "testing": {"background_routes": 45, "ev_routes": 90, "ev_assignments": 1230}}

Stations: ['42474000', '42490303', '42496394'], synthetic locations on the original network.
Reference goals are sampled uniformly from passenger/turn-reachable stations.
The CSS agent still selects its own station; reference goal lists do not override it.
Initial SOC is rejection-sampled N(0.30, 0.05^2), restricted to [0.20, 0.40],
rounded to six decimals. Base seed 20261107; SHA256-derived per-EV RNG.
Every JSON list follows actual EV XML order; ev_assignments.csv records the mapping.
See generation_protocol.json for exact reproduction and data_manifest.csv for hashes.

Audit: python verify_dataset.py --root .
Regenerate into an empty directory:
python prepare_dataset.py --source <RP dataset> --output <empty directory> --stations 42474000 42490303 42496394

Source files are preserved. No training has been started by dataset preparation.
