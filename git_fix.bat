@echo off
echo ===================================================
echo  GRAPH-FINANCE GIT FIX SCRIPT
echo ===================================================
echo.
echo Step 1: Initializing fresh git repo...
git init
echo.
echo Step 2: Setting branch to main...
git checkout -b main
echo.
echo Step 3: Adding remote origin...
git remote add origin https://github.com/Alok-Fusion/graph-finance.git
echo.
echo Step 4: Staging all project files (excl. .venv, *.pt, *.npy)...
git add -A
echo.
echo Step 5: Committing...
git commit -m "Add complete Graph-Finance project: GNN model, training, simulation, visualization, and documentation"
echo.
echo Step 6: Force pushing to GitHub (overwriting old history)...
git push --force origin main
echo.
echo ===================================================
echo  DONE! Check output above for any errors.
echo ===================================================
pause
