@echo off
REM ═══════════════════════════════════════════════════════════════════════
REM  setup_gpu.bat — Install CUDA-enabled PyTorch for RTX 4050
REM  Run this ONCE inside your venv before running train_model.py
REM
REM  RTX 4050 supports CUDA 12.x — we use the cu121 wheel index.
REM  After this, `python check_gpu.py` should show your GPU name.
REM ═══════════════════════════════════════════════════════════════════════

echo.
echo  ╔═══════════════════════════════════════════════════════╗
echo  ║   Market Prediction AI — GPU Setup (RTX 4050)        ║
echo  ║   Installing CUDA 12.6 PyTorch                       ║
echo  ╚═══════════════════════════════════════════════════════╝
echo.

REM Step 1: uninstall CPU-only torch if present
echo [1/3] Removing CPU-only PyTorch (if installed)...
pip uninstall torch torchvision torchaudio -y 2>nul

REM Step 2: install CUDA 12.6 PyTorch
echo.
echo [2/3] Installing PyTorch 2.x with CUDA 12.6 support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

REM Step 3: install remaining dependencies
echo.
echo [3/3] Installing project dependencies...
pip install -r requirements.txt

REM Verify
echo.
echo ─────────────────────────────────────────────────────────
echo  Verifying GPU detection...
python check_gpu.py

echo.
echo  Setup complete. You can now run: python train_model.py
echo ─────────────────────────────────────────────────────────
pause
