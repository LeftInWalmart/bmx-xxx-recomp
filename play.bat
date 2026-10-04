@echo off
rem Launch the recompiled BMX XXX Bevy app (builds first if needed).
rem The disc image defaults to Downloads\Emus\GAMECUBE\BMX XXX (USA).ciso; set BMX_DISC to use another .iso/.ciso.
cd /d "%~dp0rust"
cargo run -p bmx-bevy %*
if errorlevel 1 pause
