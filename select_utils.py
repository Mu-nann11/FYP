from pathlib import Path

cycle1_path = Path("D:\Munan_FYP\TMA2_BC081116d\TMA2_BC081116d\Cycle1")
cycle2_path = Path("D:\Munan_FYP\TMA2_BC081116d\TMA2_BC081116d\Cycle2")

cycle1_blocks = [p.name for p in cycle1_path.iterdir() if p.is_dir()]
cycle2_blocks = [p.name for p in cycle2_path.iterdir() if p.is_dir()]

print("Cycle1 存在的 blocks:", cycle1_blocks)
print("Cycle2 存在的 blocks:", cycle2_blocks)