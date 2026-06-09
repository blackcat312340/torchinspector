"""Colab path diagnostic — figure out what's going on."""
import sys, os, subprocess

print("=== CWD ===")
print(os.getcwd())

print("\n=== sys.path ===")
for p in sys.path:
    if p:
        print(f"  {p}")

print("\n=== pip show torchinspector ===")
r = subprocess.run([sys.executable, "-m", "pip", "show", "torchinspector"],
                   capture_output=True, text=True)
print(r.stdout or r.stderr)

print("\n=== /content/ contents ===")
for f in os.listdir('/content'):
    if 'torch' in f.lower() or 'tsrc' in f.lower():
        full = f'/content/{f}'
        has_init = os.path.exists(f'{full}/__init__.py')
        print(f"  {full}/  (__init__.py={'YES' if has_init else 'NO'})")

print("\n=== /content/torchinspector/ contents ===")
repo = '/content/torchinspector'
if os.path.exists(repo):
    for root, dirs, files in os.walk(repo):
        level = root.replace(repo, '').count(os.sep)
        if level <= 2:
            indent = '  ' * level
            name = os.path.basename(root) or repo
            print(f"{indent}{name}/")
    print("\n  __init__.py locations:")
    for root, dirs, files in os.walk(repo):
        if '__init__.py' in files:
            print(f"    {root}/__init__.py")

print("\n=== Try import ===")
try:
    import torchinspector
    print(f"  file: {torchinspector.__file__}")
    print(f"  path: {torchinspector.__path__}")
    from torchinspector import Inspector
    print("  Inspector: OK")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n=== site-packages torchinspector ===")
for p in sys.path:
    if 'site-packages' in p or 'dist-packages' in p:
        target = os.path.join(p, 'torchinspector')
        if os.path.exists(target):
            print(f"  {target}/")
            print(f"    __init__.py: {os.path.exists(os.path.join(target, '__init__.py'))}")
