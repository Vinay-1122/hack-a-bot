import ast
import re
from typing import List, Tuple, Set

class CodeValidator:
    """Validates generated code for security and safety before execution"""
    
    # Allowed imports - only safe libraries
    ALLOWED_IMPORTS = {
        'pandas', 'pd', 'numpy', 'np', 'plotly', 'plotly.express', 'plotly.graph_objects', 
        'plotly.subplots', 'plotly.io', 'math', 'datetime', 'json', 're'
    }
    
    # Dangerous imports that should never be allowed
    DANGEROUS_IMPORTS = {
        'os', 'sys', 'subprocess', 'socket', 'urllib', 'requests', 'http', 'ftplib',
        'smtplib', 'pickle', 'eval', 'exec', 'compile', '__import__', 'importlib',
        'builtins', 'threading', 'multiprocessing', 'ctypes', 'platform', 'shutil'
    }
    
    # Dangerous function calls
    DANGEROUS_FUNCTIONS = {
        'eval', 'exec', 'compile', '__import__', 'open', 'input', 'raw_input',
        'file', 'execfile', 'reload', 'vars', 'locals', 'globals', 'dir',
        'hasattr', 'getattr', 'setattr', 'delattr'
    }
    
    # Dangerous keywords/patterns
    DANGEROUS_PATTERNS = [
        r'__.*__',  # Dunder methods
        r'\.system\(',  # System calls
        r'\.popen\(',  # Process opening
        r'\.call\(',   # Subprocess calls
        r'\.run\(',    # Subprocess run
        r'with open\(',  # File operations
        r'open\(',     # File operations
        r'file\(',     # File operations
        r'import\s+os', # OS import
        r'from\s+os',   # OS import
    ]
    
    def validate_code(self, code: str) -> Tuple[bool, List[str]]:
        """
        Validate code for security issues
        
        Args:
            code: Python code to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        try:
            # Parse the AST
            tree = ast.parse(code)
            
            # Check imports
            import_errors = self._check_imports(tree)
            errors.extend(import_errors)
            
            # Check function calls
            function_errors = self._check_function_calls(tree)
            errors.extend(function_errors)
            
            # Check for dangerous patterns
            pattern_errors = self._check_dangerous_patterns(code)
            errors.extend(pattern_errors)
            
            # Check for file operations
            file_errors = self._check_file_operations(code)
            errors.extend(file_errors)
            
            return len(errors) == 0, errors
            
        except SyntaxError as e:
            return False, [f"Syntax error in generated code: {str(e)}"]
        except Exception as e:
            return False, [f"Code validation error: {str(e)}"]
    
    def _check_imports(self, tree: ast.AST) -> List[str]:
        """Check for dangerous imports"""
        errors = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split('.')[0]
                    if module_name in self.DANGEROUS_IMPORTS:
                        errors.append(f"Dangerous import detected: {alias.name}")
                    elif module_name not in self.ALLOWED_IMPORTS:
                        errors.append(f"Unauthorized import detected: {alias.name}")
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split('.')[0]
                    if module_name in self.DANGEROUS_IMPORTS:
                        errors.append(f"Dangerous import detected: from {node.module}")
                    elif module_name not in self.ALLOWED_IMPORTS:
                        errors.append(f"Unauthorized import detected: from {node.module}")
        
        return errors
    
    def _check_function_calls(self, tree: ast.AST) -> List[str]:
        """Check for dangerous function calls"""
        errors = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = None
                
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                
                if func_name in self.DANGEROUS_FUNCTIONS:
                    errors.append(f"Dangerous function call detected: {func_name}")
        
        return errors
    
    def _check_dangerous_patterns(self, code: str) -> List[str]:
        """Check for dangerous regex patterns"""
        errors = []
        
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                errors.append(f"Dangerous pattern detected: {pattern}")
        
        return errors
    
    def _check_file_operations(self, code: str) -> List[str]:
        """Check for unauthorized file operations"""
        errors = []
        
        # Allow only specific file operations
        allowed_file_ops = [
            "fig.write_html('/tmp/output.html'",
            "fig.write_html(\"/tmp/output.html\"",
            "pd.read_csv('/tmp/df",
            "pd.read_csv(\"/tmp/df"
        ]
        
        # Check for file operations
        file_patterns = [
            r'\.write_html\(',
            r'\.to_csv\(',
            r'\.to_excel\(',
            r'\.to_json\(',
            r'\.read_csv\(',
            r'\.read_excel\(',
            r'\.read_json\(',
        ]
        
        for pattern in file_patterns:
            matches = re.finditer(pattern, code)
            for match in matches:
                # Get the full line containing the match
                lines = code.split('\n')
                for line in lines:
                    if pattern.replace('\\', '') in line:
                        # Check if it's an allowed operation
                        if not any(allowed_op in line for allowed_op in allowed_file_ops):
                            if 'write_html' not in line or '/tmp/output.html' not in line:
                                errors.append(f"Unauthorized file operation: {line.strip()}")
        
        return errors
    
    def sanitize_code(self, code: str) -> str:
        """
        Sanitize code by removing comments and extra whitespace
        
        Args:
            code: Raw code to sanitize
            
        Returns:
            Sanitized code
        """
        # Remove comments (but be careful with strings)
        lines = code.split('\n')
        sanitized_lines = []
        
        for line in lines:
            # Remove inline comments that are not in strings
            in_string = False
            quote_char = None
            result = []
            i = 0
            
            while i < len(line):
                char = line[i]
                
                if not in_string and char in ['"', "'"]:
                    in_string = True
                    quote_char = char
                    result.append(char)
                elif in_string and char == quote_char and (i == 0 or line[i-1] != '\\'):
                    in_string = False
                    quote_char = None
                    result.append(char)
                elif not in_string and char == '#':
                    break  # Rest of line is comment
                else:
                    result.append(char)
                
                i += 1
            
            sanitized_line = ''.join(result).rstrip()
            if sanitized_line:  # Only add non-empty lines
                sanitized_lines.append(sanitized_line)
        
        return '\n'.join(sanitized_lines) 