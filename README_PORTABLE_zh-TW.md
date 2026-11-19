# 在另一台 Windows 電腦執行 CSS

本 repository 已包含 CSS_refactor 程式、舊 `env` / 新 `new_env` 所需路網與背景/EV/goal/power 資料，及兩環境的 Proposed、DQN 路由權重。不需要另一份 RP 專案，也不需重訓 RP。

## 1. 取得程式與環境

沿用已成功執行 RP 的 `OSM_W` conda 環境即可，保留其已驗證能使用 Blackwell GPU 的 PyTorch，不要用舊 `environment.yml` 覆蓋它。`environment.yml` 是原電腦歷史快照（PyTorch 2.3.1/CUDA 12.1），不是新設備的建議安裝檔。

在想放專案的資料夾開啟 PowerShell：

```powershell
conda activate OSM_W
git clone https://github.com/WangJimm/CSS.git
Set-Location CSS
python -m pip install -r requirements.txt
$env:SUMO_HOME = python -c "import pathlib,sumo; print(pathlib.Path(sumo.__file__).resolve().parent)"
$env:SUMO_BINARY = Join-Path $env:SUMO_HOME 'bin\sumo.exe'
$env:PATH = (Join-Path $env:SUMO_HOME 'bin') + ';' + $env:PATH
$env:PYTHONHASHSEED = '42'
python -m scripts.run_experiment --env new_env --check-data
python -m scripts.check_route_models
```

本 repository 已設為公開，可直接 clone。已經 clone 此 CSS repository 時，在其目錄執行 `git pull --ff-only`；若有本機修改，先保留修改再更新。**不要在 new_rp 資料夾 pull 期待取得 CSS。**

## 2. 四份路由模型

| CSS 環境 | `--route-mode` | RP 方法 | validation 選定 episode | 預設檔案 |
|---|---|---|---:|---|
| `env` | `dqngnn` | Proposed | 2400 | `route_planning_agent/DQN/env/dqngnn_best_better_ratio.pt` |
| `env` | `dqn` | DQN tls_time | 2800 | `route_planning_agent/DQN/env/dqn_best_better_ratio.pt` |
| `new_env` | `dqngnn` | Proposed | 2000 | `route_planning_agent/DQN/new_env/dqngnn_best_better_ratio.pt` |
| `new_env` | `dqn` | DQN tls_time | 2200 | `route_planning_agent/DQN/new_env/dqn_best_better_ratio.pt` |

檔名 `best_better_ratio` 沿用 CSS 歷史名稱；本次實際來源是上述 validation 選定的 `*_best_arrival.pt`，不是重新以 testing 選模。模型轉成純 tensor + 一般 dict 的 `rp_tensors_v1` 可攜格式，網路權重逐 tensor 完全相同；不帶 optimizer/replay，因為 CSS 只固定使用 RP 模型。每個資料夾的 `models_manifest.json` 記錄原檔 hash、輸出 hash、路網 hash、節點順序、輸入 schema 與訓練來源。

DQN 目前正確使用「目前節點＋目標節點＋號誌＋simulation time/3600」；舊環境輸入 239 維、新環境 767 維。Proposed 保留 `static_occupancy` GNN 特徵。`--route-mode dqngnn` 就是這批 Proposed 模型的 CSS 名稱。

`--checkpoint` 是正在訓練/評估的 **CSS 選站模型**，不是 RP 模型；`--route-checkpoint` 才是 RP 模型。使用上表預設路由模型不必另外指定 `--route-checkpoint`。

## 3. 四種執行組合（以 LinUCB 選站為例）

每次選一條執行；沒有在上傳時自動開始完整訓練。

```powershell
# 新環境 + Proposed RP
python -m scripts.run_experiment --env new_env --agent linucb --route-mode dqngnn --mode train --episodes 1000 --seed 42

# 新環境 + DQN RP
python -m scripts.run_experiment --env new_env --agent linucb --route-mode dqn --mode train --episodes 1000 --seed 42

# 舊環境 + Proposed RP：明確指定 CSS 模型儲存位置
python -m scripts.run_experiment --env env --agent linucb --route-mode dqngnn --mode train --episodes 1000 --seed 42 --checkpoint parameters/env/linucb_route_dqngnn.pt

# 舊環境 + DQN RP
python -m scripts.run_experiment --env env --agent linucb --route-mode dqn --mode train --episodes 1000 --seed 42 --checkpoint parameters/env/linucb_route_dqn.pt
```

改成 `--agent sacd` 等選項可以更換 CSS 選站演算法，路由模式獨立。新環境預設 CSS 模型在 `parameters/new_env/<agent>_route_<route-mode>.pt`，結果在 `outputs/new_env/`；舊環境結果仍在 `outputs/`，建議各次執行後保存輸出，以免同名歷史報表覆寫。評估 CSS 模型使用 `--mode eval --checkpoint <CSS模型路徑>`。

## 4. 保留的實驗設定

- CSS `new_env` 仍是完整 420 training / 90 validation / 90 testing 場景；testing 1905 EV，沒有套用 RP remove3。CS agent 在出發時自己選站，goal 檔僅作參考；初始 SOC 使用已保存的 0.2–0.4 截斷常態資料。
- CSS `env` 已更新成與 new_env 相同的 70/15/15：1500/1800/2100 各 100 個背景 seed，訓練/驗證/測試為 420/90/90 個場景，對應 5740/1230/1230 輛 EV。沿用 RP 舊環境的完整切分，不套用 remove4。Validation 讀取獨立的 validation_data，不再讀 testing 子集。兩種 EV 比例和相同生成 seed 不跨分區。
- Old env 初始 SOC 改用與 new_env 相同的截斷常態 N(0.30, 0.05²)，限制於 [0.20, 0.40]，固定逐 EV seed；可達站點中隨機分配的 goal 僅作參考，CSS agent 仍自行選站。既有 CSS 結果屬於舊資料版本，請重新訓練以比較。路網、RP 權重、站點 action 順序及舊地圖號誌特徵均保持原樣。
- CSS 路由決策上限目前仍是 `css/experiment/config.py` 的 `MAX_DECISIONS_PER_EV=50`，不是 RP testing 的 80；這次只銜接模型，沒有改動訓練預算。CSS 的任務、充電站目標、路由控制時機與排隊互動不同，不保證重現 RP 的抵達率/成本。
- 額外保留舊環境既有 eDQN 模型供既有相容測試；不是本次四份 RP 模型之一，新環境沒有配套 eDQN。

## 5. 已完成驗證

四份模型的原始/可攜 tensor 完全相等；同一觀察/graph 下各三組 Q 值與 RP 完全相等。目前 100 項 CSS 測試通過，四個環境×路由組合各以一輛 EV 與原背景車流完成 SUMO 檢查，均無路由失敗。這些是載入/相容性檢查，不是 CSS 完整 training 或 performance 評估。

## 6. TraCI 查詢加速（2026-11-17）

更新後沿用原指令即可，不需要額外旗標。依 `edge_feature_mode` 只收集模型使用的動態 edge 指標；例如目前 Proposed 的 `static_occupancy` 只查 occupancy，其餘未使用的動態欄位仍為零。選用指標的 EMA 更新次數與時機維持原樣；debug 模式仍收集全部指標。

固定號誌 ID/controlled links 會快取，但目前相位、切換時間與 program logic 仍即時讀取。標準 training、validation/testing、legacy Dijkstra 載入場景，以及建立 static graph 時，都會重設快取。若自訂腳本直接呼叫 `traci.load` 或修改號誌連接，請同時呼叫 `css.sumo.reset_signal_topology_cache()`。

在相同 seed 的兩個 training prefix episode 及獨立 p02（30 EV）對照中，選站/路線事件、simulation steps、reward、final LinUCB 參數逐值相同。本機 RTX 2060 的短程計時如下，僅供參考：

| 需求 / p / seed | 原版秒/episode | 新版秒/episode |
|---|---:|---:|
| 2950 / p01 / 40098 | 64.94 | 24.83 |
| 2650 / p01 / 50076 | 46.58 | 18.27 |
| 2950 / p02 / 40054 | 118.85 | 41.49 |

實際速度依電腦與場景而異；這不是完整 1000 episode 的時間保證。路由模型權重、資料切分、0.5 秒步長、決策上限與學習規則均未更改。驗證摘要在 `verification/runtime_speedup.json`。已啟動的 Python 程序不會自動套用更新，需在下次啟動時使用新版。

## 7. Old env 資料更新（2026-11-17）

兩種環境現在都採 manifest 讀取；`--env env` 即選擇更新後的舊路網資料。
每 episode 仍從訓練集有放回抽樣，`--seed 42` 可重現排程。

```powershell
python -m scripts.run_experiment --env env --check-data
python -m scripts.run_experiment --env env --agent linucb --route-mode dqngnn --mode train --episodes 1000 --seed 42 --no-train-evaluation --checkpoint parameters/env/linucb_route_dqngnn_701515_final.pt
```

資料生成與獨立驗證工具為 `scripts/prepare_css_dataset.py`、`scripts/verify_css_dataset.py`；
詳細切分、逐車 SOC/goal 對應和雜湊保存在 `env/` 的 manifest 與 protocol 檔案。
原先的 CSS 資料已保留在本機外部備份目錄，未混入目前的三個分區。

## 8. DQN RP 圖資料加速（2026-11-18）

原本的 DQN＋DQN 指令不需要更改。`--route-mode dqn` 現在明確宣告不使用路由 graph，
略過未使用的全路網 travel time／mean speed／occupancy 與候選車道特徵建構。
DQN 真正使用的 node／goal／局部號誌／simulation time、合法鄰居與路由提交維持原樣。
Visit history 使用既有更新函式，SUMO 步長、決策上限、reward、資料、RP 權重及學習頻率不變。

Proposed 與未明確宣告可省略 graph 的路由方法維持原流程。若選站模型本身使用圖觀察
（例如 paper_baseline），即使搭配 DQN RP，也保留共享 EMA 的原更新時機與圖資料收集，
避免影響其選站輸入。標準 training／validation／testing 都會套用此判斷。

同 seed 五組完整場景（new env 兩組、old env 三組含 p02）的選站／路線事件、reward、
模擬步數、replay、EPSILON、亂數狀態與最終網路參數逐值相同；另以錄得的 transition
填滿 replay，兩種環境各核對 10 次 CUDA 學習更新，loss 與兩個網路亦逐值相同。
這不是完整 250 episode 暖身後的續訓實驗。

本機 RTX2060 單次計時（含 profiler 開銷）：

| 環境／需求／比例 | 原版秒/episode | 正式加速版秒/episode |
|---|---:|---:|
| new_env／2950／p01 | 55.22 | 6.64 |
| new_env／2650／p01 | 47.03 | 5.57 |
| env／2100／p01 | 10.38 | 2.74 |
| env／1500／p01 | 7.61 | 1.65 |
| env／2100／p02 | 20.22 | 3.96 |

速度依設備及場景而異；以上不是另一台電腦或完整 1000 episode 的時間保證。
驗證紀錄：`verification/dqn_route_speedup.json`。已啟動的 Python 程序不會自動使用新程式，
更新在下次啟動時生效。

## 9. Testing 參考策略開關（2026-11-19）

在既有 testing 指令加上 `--no-eval-references`，只測指定的 CSS agent 與 RP 組合，
略過額外的「最短等待選站」及「最近充電站選站」模擬。例如：

```powershell
python -m scripts.run_experiment --env new_env --agent linucb --route-mode dqngnn --mode eval --seed 42 --checkpoint outputs/all_new_env/linucb_route_dqngnn_final.pt --no-eval-references
```

`--env env`、四種選站方法、所有 RP 模式及 `--higher-evaluation` 都可使用這個開關。
paper_baseline＋Dijkstra 自己的模型測試仍會執行；略過的是額外參考策略。
不加開關就維持原本三種 rollout。開關只影響 `--mode eval` 或 `both` 的最後 testing，
不改 training 或訓練期間 validation；停用後者請用 `--no-train-evaluation`。

JSON 保留指定模型的完整 `methods.agent`，未執行的 `methods.min_waiting` 與
`methods.shortest` 為 `null`，並在 `evaluation_options` 記錄開關；終端顯示 skipped，
不會把略過的策略顯示成成本為零。標準 testing 每個方法仍讀取完整 90 場景，
單環境四種方法由 1080 次模擬減為 360 次，實際時間依各場景而異。
