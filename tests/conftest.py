"""让测试文件能`from core.memory.exploration_memory import ...`这样按子包导入——
core/下每个文件互相import时都是这个约定（各自在文件头把项目根目录插入
sys.path），这里在测试侧统一做一次，不用每个测试文件重复插入。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
