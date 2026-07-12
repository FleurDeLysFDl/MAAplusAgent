"""让测试文件能直接`import exploration_memory`之类的裸模块名——core/下的每个
文件互相import时都是这个约定（各自在文件头`sys.path.insert(0, 当前目录)`），
这里在测试侧统一做一次，不用每个测试文件重复插入。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
