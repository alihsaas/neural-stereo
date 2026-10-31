@echo off
cd /d "%~dp0"
uv run --locked --extra neural python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || goto fail
uv run --locked --extra neural pytest -q --basetemp=.pytest-tmp || goto fail
uv run --locked --extra neural python scripts/download_data.py || goto fail
uv run --locked --extra neural python scripts/run_compression_study.py || goto fail
uv run --locked --extra neural python scripts/run_learned_study.py || goto fail
echo.
echo Finished. Results are in results\results.csv, results\results_flexglue.csv and the figures next to them.
pause
exit /b 0
:fail
echo.
echo A step failed. Read the message above.
pause
exit /b 1
