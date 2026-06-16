Set-Location "C:\Users\ylchen\workspace\timesfm_finetuning"
$py = "C:\Users\ylchen\workspace\timesfm\.venv\Scripts\python.exe"
$model = "C:\Users\ylchen\workspace\timesfm_finetuning\model"
$env:PYTHONUNBUFFERED = "1"

# Training data cutoff: Run 4 was trained 2026-02-20; eval covers data since that date.
$trainEnd = "2026-02-20"
$evalStart = "2026-02-20"

# Prior results embedded for comparison in each log
$prior = @"

PRIOR RESULTS REFERENCE
-----------------------
US  baseline: ~8.7% 120d MAPE (6 eval tickers: NVDA,UNH,GS,CVX,KO,BA)
    Run4 finetuned: ~7.0% overall (-1.0pp, 5/6 tickers improved)
    exp_11 placeholder: ~7.7%

JAP baseline: 8035.T=34.943%, 6902.T=2.883%, 4661.T=19.627%, overall=19.151%
    run1 regressed: overall=20.318% (-1.17pp)
    run_glia_t2_r2a target: beat baseline (stride=64, hdecay=1.0)

EUR baseline: UL=3.978%, EADSF=4.194%, VWAGY=5.930%, overall=4.701%
    run1 improved: UL=3.865%, EADSF=4.037%, VWAGY=5.796%, overall=4.566% (+0.135pp)
    run_glia_v2_1a target: beat run1 (stride=64)

CHN baseline: 0388.HK=4.052%, 2382.HK=21.322%, 1177.HK=31.356%, overall=18.910%
    run1 regressed: 0388.HK=7.122%, 2382.HK=26.265%, 1177.HK=37.981%, overall=23.789%
    run_glia_2a target: beat baseline (stride=64, train_long only)
"@

# ── US ────────────────────────────────────────────────────────────────────────
Write-Output "[$(Get-Date -Format 'HH:mm:ss')] === US run4 — TRAINING ===" | Tee-Object -FilePath "finetune_logs\us_run4.log"
& $py finetune_timesfm.py `
  --tickers "^GSPC,AAPL,JNJ,JPM,XOM,PG,CAT,NEE,AMT,WMT" `
  --cache-dir "C:\Users\ylchen\workspace\research_group\data\us" `
  --save-dir finetune_usa/checkpoints `
  --model-id $model `
  --epochs 50 --batch-size 8 --lr 1e-4 `
  --optimizer adamw --weight-decay 0.01 `
  --freeze-layers 17 --accumulation-steps 16 `
  --warmup-epochs 3 --early-stopping 15 `
  --max-context 512 --horizon 128 `
  --train-end-date $trainEnd `
  --no-epoch-ckpts 2>&1 | Tee-Object -Append -FilePath "finetune_logs\us_run4.log"
Write-Output "[$(Get-Date -Format 'HH:mm:ss')] US training done. Exit: $LASTEXITCODE" | Tee-Object -Append -FilePath "finetune_logs\us_run4.log"

Write-Output "[$(Get-Date -Format 'HH:mm:ss')] === US run4 — EVALUATION ===" | Tee-Object -Append -FilePath "finetune_logs\us_run4.log"
Write-Output $prior | Tee-Object -Append -FilePath "finetune_logs\us_run4.log"
& $py scripts\evaluate_forecast.py `
  --tickers "NVDA,UNH,GS,CVX,KO,BA" `
  --checkpoint finetune_usa/checkpoints/best `
  --baseline-id $model `
  --horizons "7,30,60,120" --max-context 512 `
  --eval-start-date $evalStart `
  --log-transform `
  --plot "finetune_logs\us_run4_eval.png" 2>&1 | Tee-Object -Append -FilePath "finetune_logs\us_run4.log"
Write-Output "[$(Get-Date -Format 'HH:mm:ss')] US eval done." | Tee-Object -Append -FilePath "finetune_logs\us_run4.log"

# ── JAPAN ─────────────────────────────────────────────────────────────────────
Write-Output "[$(Get-Date -Format 'HH:mm:ss')] === JAPAN run_glia_t2_r2a — TRAINING ===" | Tee-Object -FilePath "finetune_logs\japan_run_glia_t2_r2a.log"
& $py finetune_timesfm.py `
  --data-dir finetune_japan/data/train `
  --save-dir finetune_japan/checkpoints/run_glia_t2_r2a `
  --model-id $model `
  --epochs 50 --batch-size 8 --lr 1e-4 `
  --optimizer adamw --weight-decay 0.01 `
  --freeze-layers 17 --accumulation-steps 16 `
  --warmup-epochs 5 --early-stopping 15 `
  --max-context 512 --horizon 128 `
  --stride 64 --horizon-loss-decay 1.0 `
  --no-epoch-ckpts 2>&1 | Tee-Object -Append -FilePath "finetune_logs\japan_run_glia_t2_r2a.log"
Write-Output "[$(Get-Date -Format 'HH:mm:ss')] Japan training done. Exit: $LASTEXITCODE" | Tee-Object -Append -FilePath "finetune_logs\japan_run_glia_t2_r2a.log"

Write-Output "[$(Get-Date -Format 'HH:mm:ss')] === JAPAN run_glia_t2_r2a — EVALUATION ===" | Tee-Object -Append -FilePath "finetune_logs\japan_run_glia_t2_r2a.log"
Write-Output $prior | Tee-Object -Append -FilePath "finetune_logs\japan_run_glia_t2_r2a.log"
& $py scripts\evaluate_forecast.py `
  --tickers "8035.T,6902.T,4661.T" `
  --checkpoint finetune_japan/checkpoints/run_glia_t2_r2a/best `
  --baseline-id $model `
  --horizons "120" --max-context 512 `
  --eval-start-date $evalStart `
  --log-transform `
  --plot "finetune_logs\japan_run_glia_t2_r2a_eval.png" 2>&1 | Tee-Object -Append -FilePath "finetune_logs\japan_run_glia_t2_r2a.log"
Write-Output "[$(Get-Date -Format 'HH:mm:ss')] Japan eval done." | Tee-Object -Append -FilePath "finetune_logs\japan_run_glia_t2_r2a.log"

# ── EUROPE ────────────────────────────────────────────────────────────────────
Write-Output "[$(Get-Date -Format 'HH:mm:ss')] === EUROPE run_glia_v2_1a — TRAINING ===" | Tee-Object -FilePath "finetune_logs\europe_run_glia_v2_1a.log"
& $py finetune_timesfm.py `
  --data-dir finetune_europe/data/train `
  --save-dir finetune_europe/checkpoints/run_glia_v2_1a `
  --model-id $model `
  --epochs 50 --batch-size 8 --lr 1e-4 `
  --optimizer adamw --weight-decay 0.01 `
  --freeze-layers 17 --accumulation-steps 16 `
  --warmup-epochs 5 --early-stopping 15 `
  --max-context 512 --horizon 128 `
  --stride 64 `
  --no-epoch-ckpts 2>&1 | Tee-Object -Append -FilePath "finetune_logs\europe_run_glia_v2_1a.log"
Write-Output "[$(Get-Date -Format 'HH:mm:ss')] Europe training done. Exit: $LASTEXITCODE" | Tee-Object -Append -FilePath "finetune_logs\europe_run_glia_v2_1a.log"

Write-Output "[$(Get-Date -Format 'HH:mm:ss')] === EUROPE run_glia_v2_1a — EVALUATION ===" | Tee-Object -Append -FilePath "finetune_logs\europe_run_glia_v2_1a.log"
Write-Output $prior | Tee-Object -Append -FilePath "finetune_logs\europe_run_glia_v2_1a.log"
& $py scripts\evaluate_forecast.py `
  --tickers "UL,EADSF,VWAGY" `
  --checkpoint finetune_europe/checkpoints/run_glia_v2_1a/best `
  --baseline-id $model `
  --horizons "120" --max-context 512 `
  --eval-start-date $evalStart `
  --log-transform `
  --plot "finetune_logs\europe_run_glia_v2_1a_eval.png" 2>&1 | Tee-Object -Append -FilePath "finetune_logs\europe_run_glia_v2_1a.log"
Write-Output "[$(Get-Date -Format 'HH:mm:ss')] Europe eval done." | Tee-Object -Append -FilePath "finetune_logs\europe_run_glia_v2_1a.log"

# ── CHINA ─────────────────────────────────────────────────────────────────────
Write-Output "[$(Get-Date -Format 'HH:mm:ss')] === CHINA run_glia_2a — TRAINING ===" | Tee-Object -FilePath "finetune_logs\china_run_glia_2a.log"
& $py finetune_timesfm.py `
  --data-dir finetune_china/data/train_long `
  --save-dir finetune_china/checkpoints/run_glia_2a `
  --model-id $model `
  --epochs 50 --batch-size 8 --lr 1e-4 `
  --optimizer adamw --weight-decay 0.01 `
  --freeze-layers 17 --accumulation-steps 16 `
  --warmup-epochs 5 --early-stopping 15 `
  --max-context 512 --horizon 128 `
  --stride 64 `
  --no-epoch-ckpts 2>&1 | Tee-Object -Append -FilePath "finetune_logs\china_run_glia_2a.log"
Write-Output "[$(Get-Date -Format 'HH:mm:ss')] China training done. Exit: $LASTEXITCODE" | Tee-Object -Append -FilePath "finetune_logs\china_run_glia_2a.log"

Write-Output "[$(Get-Date -Format 'HH:mm:ss')] === CHINA run_glia_2a — EVALUATION ===" | Tee-Object -Append -FilePath "finetune_logs\china_run_glia_2a.log"
Write-Output $prior | Tee-Object -Append -FilePath "finetune_logs\china_run_glia_2a.log"
& $py scripts\evaluate_forecast.py `
  --tickers "0388.HK,2382.HK,1177.HK" `
  --checkpoint finetune_china/checkpoints/run_glia_2a/best `
  --baseline-id $model `
  --horizons "120" --max-context 512 `
  --eval-start-date $evalStart `
  --log-transform `
  --plot "finetune_logs\china_run_glia_2a_eval.png" 2>&1 | Tee-Object -Append -FilePath "finetune_logs\china_run_glia_2a.log"
Write-Output "[$(Get-Date -Format 'HH:mm:ss')] China eval done." | Tee-Object -Append -FilePath "finetune_logs\china_run_glia_2a.log"

# ── DONE ──────────────────────────────────────────────────────────────────────
Write-Output "[$(Get-Date -Format 'HH:mm:ss')] === ALL REGIONS COMPLETE ===" | Tee-Object -FilePath "finetune_logs\all_regions_done.log"
