Set-Location "C:\Users\ylchen\workspace\timesfm_finetuning"
$py = "C:\Users\ylchen\workspace\timesfm\.venv\Scripts\python.exe"
$model = "C:\Users\ylchen\workspace\timesfm_finetuning\model"
$env:PYTHONUNBUFFERED = "1"

$trainEnd = "2026-02-20"
$evalStart = "2026-02-20"

$prior = @"

PRIOR RESULTS REFERENCE
-----------------------
US  baseline: ~8.7% 120d MAPE (6 eval tickers: NVDA,UNH,GS,CVX,KO,BA)
    Run4 finetuned (original): ~7.0% overall (-1.0pp, 5/6 tickers improved)
    Run4 w3 warmup (this session): 8.9% overall (+1.8pp regressed)
"@

Write-Output "[$(Get-Date -Format 'HH:mm:ss')] === US run4_w10 — TRAINING (warmup=10) ===" | Tee-Object -FilePath "finetune_logs\us_run4_w10.log"
& $py finetune_timesfm.py `
  --tickers "^GSPC,AAPL,JNJ,JPM,XOM,PG,CAT,NEE,AMT,WMT" `
  --cache-dir "C:\Users\ylchen\workspace\research_group\data\us" `
  --save-dir finetune_usa/checkpoints `
  --model-id $model `
  --epochs 50 --batch-size 8 --lr 1e-4 `
  --optimizer adamw --weight-decay 0.01 `
  --freeze-layers 17 --accumulation-steps 16 `
  --warmup-epochs 10 --early-stopping 15 `
  --max-context 512 --horizon 128 `
  --train-end-date $trainEnd `
  --no-epoch-ckpts 2>&1 | Tee-Object -Append -FilePath "finetune_logs\us_run4_w10.log"
Write-Output "[$(Get-Date -Format 'HH:mm:ss')] US training done. Exit: $LASTEXITCODE" | Tee-Object -Append -FilePath "finetune_logs\us_run4_w10.log"

Write-Output "[$(Get-Date -Format 'HH:mm:ss')] === US run4_w10 — EVALUATION ===" | Tee-Object -Append -FilePath "finetune_logs\us_run4_w10.log"
Write-Output $prior | Tee-Object -Append -FilePath "finetune_logs\us_run4_w10.log"
& $py scripts\evaluate_forecast.py `
  --tickers "NVDA,UNH,GS,CVX,KO,BA" `
  --checkpoint finetune_usa/checkpoints/best `
  --baseline-id $model `
  --horizons "7,30,60,120" --max-context 512 `
  --eval-start-date $evalStart `
  --log-transform `
  --plot "finetune_logs\us_run4_w10_eval.png" 2>&1 | Tee-Object -Append -FilePath "finetune_logs\us_run4_w10.log"
Write-Output "[$(Get-Date -Format 'HH:mm:ss')] US eval done." | Tee-Object -Append -FilePath "finetune_logs\us_run4_w10.log"
