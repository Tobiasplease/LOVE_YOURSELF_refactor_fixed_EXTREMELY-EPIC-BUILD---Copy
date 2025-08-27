#!/usr/bin/env python3
"""
Comprehensive dependency graph analyzer for Python projects.
Recursively traces all imports starting from machine.py
"""

import ast
import os
import sys
from collections import defaultdict, deque
from typing import Dict, Set, List, Tuple, Optional


class DependencyAnalyzer:
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.all_imports: Dict[str, List[str]] = defaultdict(list)
        self.python_files: Set[str] = set()
        self.orphaned_files: Set[str] = set()
        self.circular_dependencies: List[List[str]] = []
        
    def extract_imports(self, file_path: str) -> List[str]:
        """Extract all import statements from a Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
            tree = ast.parse(content, filename=file_path)
        except (SyntaxError, UnicodeDecodeError, OSError):
            return []
        
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
                    
        return imports

    def is_local_import(self, import_name: str) -> bool:
        """Check if import refers to a local project file."""
        # Handle relative imports
        if import_name.startswith('.'):
            return True
            
        # Replace dots with path separators
        potential_path = import_name.replace('.', os.sep)
        
        # Check for module directory with __init__.py
        full_module_path = os.path.join(self.project_root, potential_path, '__init__.py')
        if os.path.exists(full_module_path):
            return True
            
        # Check for .py file
        full_file_path = os.path.join(self.project_root, potential_path + '.py')
        if os.path.exists(full_file_path):
            return True
            
        return False

    def get_python_files(self) -> Set[str]:
        """Get all Python files in the project."""
        python_files = set()
        for root, dirs, files in os.walk(self.project_root):
            # Skip common non-source directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, self.project_root).replace(os.sep, '/')
                    python_files.add(relative_path)
        return python_files

    def build_dependency_graph(self):
        """Build complete dependency graph."""
        self.python_files = self.get_python_files()
        
        for file_path in self.python_files:
            full_path = os.path.join(self.project_root, file_path.replace('/', os.sep))
            imports = self.extract_imports(full_path)
            
            for imp in imports:
                self.all_imports[file_path].append(imp)
                if self.is_local_import(imp):
                    # Map import to actual file
                    actual_file = self.resolve_import_to_file(imp)
                    if actual_file:
                        self.dependencies[file_path].add(actual_file)
                        self.reverse_dependencies[actual_file].add(file_path)

    def resolve_import_to_file(self, import_name: str) -> Optional[str]:
        """Resolve an import name to actual file path."""
        # Handle relative imports (would need current module context)
        if import_name.startswith('.'):
            return None  # Skip for now
            
        potential_path = import_name.replace('.', '/')
        
        # Check for module directory
        module_init = potential_path + '/__init__.py'
        if module_init in self.python_files:
            return module_init
            
        # Check for .py file
        py_file = potential_path + '.py'
        if py_file in self.python_files:
            return py_file
            
        return None

    def find_circular_dependencies(self):
        """Find circular dependency chains."""
        visited = set()
        rec_stack = set()
        path = []
        
        def dfs(node: str) -> bool:
            if node in rec_stack:
                # Found cycle - extract it
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                self.circular_dependencies.append(cycle)
                return True
                
            if node in visited:
                return False
                
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for dep in self.dependencies.get(node, []):
                if dfs(dep):
                    return True
                    
            path.pop()
            rec_stack.remove(node)
            return False
            
        for file in self.python_files:
            if file not in visited:
                dfs(file)

    def find_orphaned_files(self):
        """Find files that are never imported."""
        imported_files = set()
        for deps in self.dependencies.values():
            imported_files.update(deps)
        
        # machine.py is the entry point, so it's not orphaned even if not imported
        imported_files.add('machine.py')
        
        # __init__.py files are not directly imported but serve as module markers
        for file in self.python_files:
            if file.endswith('__init__.py'):
                imported_files.add(file)
                
        self.orphaned_files = self.python_files - imported_files

    def trace_dependencies_from_file(self, start_file: str, max_depth: int = 10) -> Dict[str, int]:
        """Trace all dependencies from a starting file with depth tracking."""
        visited = {}
        queue = deque([(start_file, 0)])
        
        while queue:
            file, depth = queue.popleft()
            
            if file in visited or depth > max_depth:
                continue
                
            visited[file] = depth
            
            for dep in self.dependencies.get(file, []):
                if dep not in visited:
                    queue.append((dep, depth + 1))
                    
        return visited

    def analyze(self):
        """Run complete analysis."""
        print("Building dependency graph...")
        self.build_dependency_graph()
        
        print("Finding circular dependencies...")
        self.find_circular_dependencies()
        
        print("Finding orphaned files...")
        self.find_orphaned_files()

    def generate_tree_representation(self, start_file: str = 'machine.py') -> str:
        """Generate a tree-like representation of dependencies."""
        if start_file not in self.python_files:
            return f"File {start_file} not found in project"
            
        def build_tree(file: str, visited: Set[str], depth: int = 0, max_depth: int = 8) -> List[str]:
            if depth > max_depth or file in visited:
                if file in visited:
                    return [f"{'  ' * depth}├── {file} (already seen)"]
                else:
                    return [f"{'  ' * depth}├── ... (max depth reached)"]
                    
            visited.add(file)
            
            lines = []
            deps = sorted(self.dependencies.get(file, []))
            
            for i, dep in enumerate(deps):
                is_last = (i == len(deps) - 1)
                prefix = "└── " if is_last else "├── "
                lines.append(f"{'  ' * depth}{prefix}{dep}")
                
                if dep in self.dependencies:
                    sub_lines = build_tree(dep, visited.copy(), depth + 1, max_depth)
                    lines.extend(sub_lines)
                    
            return lines
            
        visited = set()
        tree_lines = [start_file]
        tree_lines.extend(build_tree(start_file, visited))
        
        return '\n'.join(tree_lines)


def main():
    project_root = os.getcwd()
    analyzer = DependencyAnalyzer(project_root)
    analyzer.analyze()
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()
    
    print("\n=== MACHINE.PY DEPENDENCY TREE ===")
    tree = analyzer.generate_tree_representation('machine.py')
    print(tree)
    
    print(f"\n=== SUMMARY STATISTICS ===")
    print(f"Total Python files: {len(analyzer.python_files)}")
    print(f"Files with dependencies: {len([f for f in analyzer.dependencies if analyzer.dependencies[f]])}")
    print(f"Orphaned files: {len(analyzer.orphaned_files)}")
    print(f"Circular dependency chains: {len(analyzer.circular_dependencies)}")
    
    if analyzer.orphaned_files:
        print(f"\n=== ORPHANED FILES ===")
        for file in sorted(analyzer.orphaned_files):
            print(f"  {file}")
    
    if analyzer.circular_dependencies:
        print(f"\n=== CIRCULAR DEPENDENCIES ===")
        for i, cycle in enumerate(analyzer.circular_dependencies, 1):
            print(f"  Cycle {i}: {' -> '.join(cycle)}")