Set-Location "C:\Users\ylchen\workspace\timesfm_finetuning"
$py = "C:\Users\ylchen\workspace\timesfm\.venv\Scripts\python.exe"
$model = "C:\Users\ylchen\workspace\timesfm_finetuning\model"
$env:PYTHONUNBUFFERED = "1"

$trainEnd = "2026-02-20"
$evalStart = "2026-02-20"

$prior = @"

PRIOR RESULTS REFERENCE
-----------------------
US Run4 original (2026-02-20): val_loss=0.0128, overall MAPE 8.1%->7.0% (-1.0pp), 5/6 improved
  h7: -0.5pp  h30: -0.2pp  h60: -0.2pp  h120: -1.0pp
US Run4 w3 nolog (this session target): replicate above
"@

Write-Output "[$(Get-Date -Format 'HH:mm:ss')] === US run4_replicate — TRAINING (warmup=3, no log transform) ===" | Tee-Object -FilePath "finetune_logs\us_run4_replicate.log"
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
  --no-epoch-ckpts 2>&1 | Tee-Object -Append -FilePath "finetune_logs\us_run4_replicate.log"
Write-Output "[$(Get-Date -Format 'HH:mm:ss')] US training done. Exit: $LASTEXITCODE" | Tee-Object -Append -FilePath "finetune_logs\us_run4_replicate.log"

Write-Output "[$(Get-Date -Format 'HH:mm:ss')] === US run4_replicate — EVALUATION ===" | Tee-Object -Append -FilePath "finetune_logs\us_run4_replicate.log"
Write-Output $prior | Tee-Object -Append -FilePath "finetune_logs\us_run4_replicate.log"
& $py scripts\evaluate_forecast.py `
  --tickers "NVDA,UNH,GS,CVX,KO,BA" `
  --checkpoint finetune_usa/checkpoints/best `
  --baseline-id $model `
  --horizons "7,30,60,120" --max-context 512 `
  --eval-start-date $evalStart `
  --log-transform `
  --plot "finetune_logs\us_run4_replicate_eval.png" 2>&1 | Tee-Object -Append -FilePath "finetune_logs\us_run4_replicate.log"
Write-Output "[$(Get-Date -Format 'HH:mm:ss')] US eval done." | Tee-Object -Append -FilePath "finetune_logs\us_run4_replicate.log"
