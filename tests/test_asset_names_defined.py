"""上游改素材名后（2026-10-29 ARK_CHECK → ARK_CHECK_UNDERGROUND），fork 里残留的旧名字要在测试里暴露出来。"""
import ast
import builtins
from pathlib import Path
import re
import unittest
import warnings


ROOT = Path(__file__).resolve().parents[1]
CONSTANT = re.compile(r'[A-Z][A-Z0-9_]{2,}')


def module_trees():
    for path in sorted((ROOT / 'module').rglob('*.py')):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SyntaxWarning)
            yield path, ast.parse(path.read_text(encoding='utf-8'))


def assigned_names(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
    return names


class AssetNamesDefinedTest(unittest.TestCase):
    def test_explicit_asset_imports_exist(self):
        missing = []
        for path, tree in module_trees():
            for node in ast.walk(tree):
                if not (isinstance(node, ast.ImportFrom) and (node.module or '').endswith('.assets')):
                    continue
                target = ROOT / (node.module.replace('.', '/') + '.py')
                if not target.exists():
                    continue
                defined = assigned_names(ast.parse(target.read_text(encoding='utf-8')))
                missing += [f'{path.relative_to(ROOT)}: {a.name}' for a in node.names
                            if a.name != '*' and a.name not in defined]
        self.assertEqual(missing, [])

    def test_constant_names_are_defined_somewhere(self):
        # star import 进来的素材名没法按文件核对，只要求全仓库至少有一处定义
        defined, used = set(dir(builtins)), {}
        for path, tree in module_trees():
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    defined.add(node.id)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    defined.add(node.name)
                elif isinstance(node, ast.arg):
                    defined.add(node.arg)
                elif isinstance(node, ast.Import):
                    defined.update(a.asname or a.name.split('.')[0] for a in node.names)
                elif isinstance(node, ast.ImportFrom):
                    from_assets = (node.module or '').endswith('.assets')
                    defined.update(a.asname or a.name for a in node.names if a.asname or not from_assets)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and CONSTANT.fullmatch(node.id):
                    used.setdefault(node.id, str(path.relative_to(ROOT)))
        self.assertEqual({name: where for name, where in used.items() if name not in defined}, {})


if __name__ == '__main__':
    unittest.main()
