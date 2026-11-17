# CSS new_env 資料集

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

CSS 主程式現已支援 --env env / --env new_env 選擇場景。
新版會依 scenario_manifest.csv 選取 training 420、validation 90、testing 90 組資料，
訓練 validation 與最後 testing 分別讀取自己的資料夾，不再套用舊場景連續 seed 範圍。

在 CSS_refactor 目錄檢查資料（不啟動 SUMO、不要求路徑規劃模型）：

    python -m scripts.run_experiment --env new_env --check-data

啟動新版時可用 --route-mode dijkstra；若使用學習式路徑規劃，
請用 --route-checkpoint 指定在新路網上訓練的相容 RP 模型。
CSS 不會自動回退讀取舊 Brooklyn 路徑規劃模型。
結果輸出至 outputs/new_env，選站模型預設存至 parameters/new_env。
本次功能修改只以靜態資料檢查、模型載入／單次推論與 mock 測試驗證，沒有啟動正式模擬或訓練。

CSS 由 agent 在出發時選擇充電站，隨機 goal 檔不覆蓋 agent 動作；
goal 檔保留作固定隨機基準或後續串接。初始 SOC 則直接讀取本資料集 power 檔。
網路與 osm.sumocfg 沿用 RP；config 未指定 route-files，由 loader 指定所選場景的車流。

重建（使用空的新目錄，避免覆蓋）：

    python prepare_dataset.py --source <RP_refactor/new_env> --output <empty_new_directory>

完整生成規則見 generation_protocol.json；完整查核結果見 verification.json。

靜態驗證（需能讀取原 RP 資料來源）：

    python verify_dataset.py --root .

生成器測試：

    python -B test_generation.py
