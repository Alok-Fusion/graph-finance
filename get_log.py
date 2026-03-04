import subprocess
try:
    log_out = subprocess.check_output(['git', 'log', '--oneline', '-n', '15'], text=True)
    with open('githistory.txt', 'w', encoding='utf-8') as f:
        f.write(log_out)
    print("Git history written to githistory.txt")
except Exception as e:
    print(f"Error: {e}")
