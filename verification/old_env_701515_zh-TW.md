# CSS old env 70／15／15 更新

已將 CSS_refactor/env 更新成 RP 舊環境的完整 70／15／15 分配。
需求 1500、1800、2100 各 100 個背景 seed；每個背景搭配 p01／p02。
沿用 RP 的既有切分與路線，未套用 remove4；背景／EV 來源均保留。

| 分區 | 每種需求 seed | 背景車流檔 | EV 場景／goal／power 檔（各） | EV 筆數 |
|---|---:|---:|---:|---:|
| training | 70 | 210 | 420 | 5740 |
| validation | 15 | 45 | 90 | 1230 |
| testing | 15 | 45 | 90 | 1230 |

同一背景的兩種 EV 比例不跨分區；跨需求共用的數字 seed 亦不跨分區。
三組分區沒有生成 seed 或路線檔雜湊重複。

充電站仍為 42496394、42490303、42474000，並保留此 action 順序。
每輛 EV 的參考 goal 從可合法抵達的站點等機率抽選，600 組全部與 RP 原 goal 相符。
CSS agent 仍自行選站，參考 goal 不覆蓋模型 action。
起始 SOC 按 new env 規則生成：N(0.30, 0.05²) 拒絕抽樣至 [0.20,0.40]，
四捨五入至六位小數；base seed 20261107，再依需求／比例／背景 seed／真實 EV ID
建立 SHA256 個別 seed。8200 筆的平均 0.3001513、樣本 SD 0.0443323。
逐車對應與生成規則在 env/ev_assignments.csv、generation_protocol.json。

程式以 --env env 選新版舊路網資料。training 以有放回方式從 420 場景抽樣；
validation 改讀獨立 validation_data 共 90 場景，testing 亦為獨立 90 場景。
舊路網的 Proposed 號誌特徵仍沿用舊模型 convention；不因 manifest 切分切換。
路網、RP 權重、步長與決策上限未改。既有 CSS 訓練結果仍屬之前的資料版本。

檢查與訓練（在 CSS_refactor 目錄執行）：

```powershell
python -m scripts.run_experiment --env env --check-data
python -m scripts.run_experiment --env env --agent linucb --route-mode dqngnn --mode train --episodes 1000 --seed 42 --no-train-evaluation --checkpoint parameters/env/linucb_route_dqngnn_701515_final.pt
```

驗證：95 項單元測試；900 份路線 XML／合法轉向、600 組 goal/SOC 可重現性／
可達性／ID 順序／雜湊；三種路由各單 EV 模擬皆無失敗；另完成一個 11 EV
完整 training episode，無路由失敗、無 validation，最終參數正常存檔。
這些是資料與執行檢查，並非完整 1000 episode 的訓練效能評估。
新版資料與讀取程式隨此 repository 提供；原資料備份只保留於本機。

原資料備份：
C:\Users\Master\Documents\Codex\2026-10-21\new-chat\outputs\css_old_env_701515_2026-11-17\legacy_env_backup

完整機器可讀紀錄：verification_summary.json。
