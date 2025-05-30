import ast
import logging
from typing import Set, List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SecurityViolation:
    """Represents a security violation found in code."""
    type: str
    message: str
    line_number: Optional[int] = None
    severity: str = "high"


@dataclass
class CodeAnalysisResult:
    """Result of code analysis."""
    is_safe: bool
    violations: List[SecurityViolation]
    warnings: List[str]
    metadata: Dict[str, Any]


class CodeAnalyzer:
    """Static code analyzer for security and safety checks."""
    
    def __init__(self):
        # Forbidden imports that pose security risks
        self.forbidden_imports = {
            'os', 'subprocess', 'requests', 'urllib', 'urllib2', 'urllib3',
            'socket', 'http', 'https', 'ftplib', 'smtplib', 'telnetlib',
            'shutil', 'glob', 'tempfile', 'pickle', 'cPickle', 'marshal',
            'imp', 'importlib', '__import__', 'compile', 'eval', 'exec',
            'execfile', 'input', 'raw_input', 'file', 'open'
        }
        
        # Forbidden function calls
        self.forbidden_functions = {
            'eval', 'exec', 'execfile', 'compile', '__import__',
            'getattr', 'setattr', 'delattr', 'hasattr', 'vars', 'dir',
            'globals', 'locals', 'input', 'raw_input'
        }
        
        # Allowed imports for data science and plotting
        self.allowed_imports = {
            'pandas', 'numpy', 'plotly', 'matplotlib', 'seaborn',
            'dash', 'json', 'datetime', 'time', 'math', 'statistics',
            'collections', 'itertools', 'functools', 'operator',
            're', 'string', 'decimal', 'fractions', 'random',
            'io', 'csv', 'warnings', 'logging'
        }
        
        # Allowed file operations (only in specific directories)
        self.allowed_file_paths = {'/app/data/', './data/', 'data/'}
    
    def analyze_code(self, code: str) -> CodeAnalysisResult:
        """
        Perform comprehensive static analysis of Python code.
        
        Args:
            code: Python code to analyze
            
        Returns:
            CodeAnalysisResult: Analysis results with safety assessment
        """
        violations = []
        warnings = []
        metadata = {}
        
        try:
            # Parse the code into an AST
            tree = ast.parse(code)
            metadata['ast_nodes'] = len(list(ast.walk(tree)))
            
            # Perform various security checks
            violations.extend(self._check_imports(tree))
            violations.extend(self._check_function_calls(tree))
            violations.extend(self._check_file_operations(tree))
            violations.extend(self._check_dynamic_execution(tree))
            violations.extend(self._check_network_operations(tree))
            violations.extend(self._check_system_calls(tree))
            
            # Check for warnings
            warnings.extend(self._check_dangerous_patterns(tree))
            warnings.extend(self._check_resource_usage(tree))
            
            # Additional metadata
            metadata.update({
                'import_count': len([n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]),
                'function_def_count': len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]),
                'class_def_count': len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]),
                'line_count': len(code.split('\n'))
            })
            
        except SyntaxError as e:
            violations.append(SecurityViolation(
                type="syntax_error",
                message=f"Syntax error in code: {e}",
                line_number=e.lineno,
                severity="critical"
            ))
        except Exception as e:
            violations.append(SecurityViolation(
                type="analysis_error",
                message=f"Error analyzing code: {e}",
                severity="high"
            ))
        
        # Determine if code is safe
        critical_violations = [v for v in violations if v.severity in ["critical", "high"]]
        is_safe = len(critical_violations) == 0
        
        logger.info(f"Code analysis complete: {'SAFE' if is_safe else 'UNSAFE'} ({len(violations)} violations)")
        
        return CodeAnalysisResult(
            is_safe=is_safe,
            violations=violations,
            warnings=warnings,
            metadata=metadata
        )
    
    def _check_imports(self, tree: ast.AST) -> List[SecurityViolation]:
        """Check for forbidden imports."""
        violations = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split('.')[0]
                    if module_name in self.forbidden_imports:
                        violations.append(SecurityViolation(
                            type="forbidden_import",
                            message=f"Forbidden import: {alias.name}",
                            line_number=node.lineno,
                            severity="high"
                        ))
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split('.')[0]
                    if module_name in self.forbidden_imports:
                        violations.append(SecurityViolation(
                            type="forbidden_import",
                            message=f"Forbidden import from: {node.module}",
                            line_number=node.lineno,
                            severity="high"
                        ))
        
        return violations
    
    def _check_function_calls(self, tree: ast.AST) -> List[SecurityViolation]:
        """Check for forbidden function calls."""
        violations = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = None
                
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                
                if func_name and func_name in self.forbidden_functions:
                    violations.append(SecurityViolation(
                        type="forbidden_function",
                        message=f"Forbidden function call: {func_name}",
                        line_number=node.lineno,
                        severity="high"
                    ))
        
        return violations
    
    def _check_file_operations(self, tree: ast.AST) -> List[SecurityViolation]:
        """Check for potentially unsafe file operations."""
        violations = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check for open() calls
                if (isinstance(node.func, ast.Name) and node.func.id == 'open') or \
                   (isinstance(node.func, ast.Attribute) and node.func.attr == 'open'):
                    
                    # Try to extract file path from arguments
                    if node.args and isinstance(node.args[0], ast.Constant):
                        file_path = node.args[0].value
                        if isinstance(file_path, str):
                            if not any(file_path.startswith(allowed) for allowed in self.allowed_file_paths):
                                violations.append(SecurityViolation(
                                    type="unsafe_file_access",
                                    message=f"File access outside allowed directories: {file_path}",
                                    line_number=node.lineno,
                                    severity="high"
                                ))
        
        return violations
    
    def _check_dynamic_execution(self, tree: ast.AST) -> List[SecurityViolation]:
        """Check for dynamic code execution."""
        violations = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = None
                
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                
                if func_name in ['eval', 'exec', 'compile']:
                    violations.append(SecurityViolation(
                        type="dynamic_execution",
                        message=f"Dynamic code execution detected: {func_name}",
                        line_number=node.lineno,
                        severity="critical"
                    ))
        
        return violations
    
    def _check_network_operations(self, tree: ast.AST) -> List[SecurityViolation]:
        """Check for network operations."""
        violations = []
        
        network_modules = {'requests', 'urllib', 'urllib2', 'urllib3', 'http', 'socket'}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                # Check for network-related method calls
                if hasattr(node.value, 'id') and node.value.id in network_modules:
                    violations.append(SecurityViolation(
                        type="network_operation",
                        message=f"Network operation detected: {node.value.id}.{node.attr}",
                        line_number=node.lineno,
                        severity="high"
                    ))
        
        return violations
    
    def _check_system_calls(self, tree: ast.AST) -> List[SecurityViolation]:
        """Check for system calls and subprocess usage."""
        violations = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    # Check for subprocess calls
                    if (hasattr(node.func.value, 'id') and 
                        node.func.value.id == 'subprocess'):
                        violations.append(SecurityViolation(
                            type="subprocess_call",
                            message=f"Subprocess call detected: subprocess.{node.func.attr}",
                            line_number=node.lineno,
                            severity="critical"
                        ))
                    
                    # Check for os system calls
                    if (hasattr(node.func.value, 'id') and 
                        node.func.value.id == 'os' and 
                        node.func.attr in ['system', 'popen', 'spawn']):
                        violations.append(SecurityViolation(
                            type="system_call",
                            message=f"System call detected: os.{node.func.attr}",
                            line_number=node.lineno,
                            severity="critical"
                        ))
        
        return violations
    
    def _check_dangerous_patterns(self, tree: ast.AST) -> List[str]:
        """Check for potentially dangerous patterns that warrant warnings."""
        warnings = []
        
        # Check for infinite loops
        for node in ast.walk(tree):
            if isinstance(node, ast.While):
                if isinstance(node.test, ast.Constant) and node.test.value is True:
                    warnings.append("Potential infinite loop detected (while True)")
        
        return warnings
    
    def _check_resource_usage(self, tree: ast.AST) -> List[str]:
        """Check for patterns that might consume excessive resources."""
        warnings = []
        
        # Check for large list comprehensions or loops
        for node in ast.walk(tree):
            if isinstance(node, ast.ListComp):
                # This is a simple heuristic - could be enhanced
                warnings.append("Large list comprehension detected - monitor memory usage")
        
        return warnings
    
    def extract_code_from_response(self, llm_response: str) -> Optional[str]:
        """
        Extract Python code from LLM response.
        
        Args:
            llm_response: Raw response from LLM
            
        Returns:
            Optional[str]: Extracted Python code or None if not found
        """
        import re
        
        # Look for code blocks marked with ```python
        python_pattern = r'```python\s*\n(.*?)\n```'
        matches = re.findall(python_pattern, llm_response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # Look for generic code blocks
        generic_pattern = r'```\s*\n(.*?)\n```'
        matches = re.findall(generic_pattern, llm_response, re.DOTALL)
        
        if matches:
            # Try to validate if it's Python code
            code = matches[0].strip()
            try:
                ast.parse(code)
                return code
            except SyntaxError:
                pass
        
        # If no code blocks found, return None
        return None 