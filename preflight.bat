@echo off
rem Headless boot check: runs the recompiled game without a window for N seconds (default 90)
rem and reports whether it renders, keeps progressing, plays audio and uses the memory card.
rem The report is written to rust\preflight_report.txt. Uses a throwaway memory card.
cd /d "%~dp0rust"
set BMX_MEMCARD=%TEMP%\bmx_preflight_card.bin
del /q "%BMX_MEMCARD%" 2>nul
cargo run -p bmx-recomp --bin preflight -- %1
set BMX_MEMCARD=
pause
