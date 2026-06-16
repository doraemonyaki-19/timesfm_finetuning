@echo off
REM Weekly mark-to-market for the board-recommended paper trade (local, durable).
REM Created for the TimesFM_PaperTrade_WeeklyMTM scheduled task.
cd /d C:\Users\ylchen\workspace\timesfm_finetuning
echo ===================================================================== >> board_recommended_test\mtm_log.txt
echo MTM run %DATE% %TIME% >> board_recommended_test\mtm_log.txt
C:\Users\ylchen\workspace\timesfm\.venv\Scripts\python.exe board_recommended_test\paper_trade.py run >> board_recommended_test\mtm_log.txt 2>&1
echo. >> board_recommended_test\mtm_log.txt
