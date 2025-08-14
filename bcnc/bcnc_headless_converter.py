#!/usr/bin/env python3
"""
Attempt to use bCNC's internal modules for headless SVG conversion
This tries to import and use bCNC's conversion logic directly
"""

import sys
import os

# === Configuration ===
base_path = "/home/jbe/Dropbox/_outputs"
svg_input = f"{base_path}/impostor-20250725_185854_00001_.png.svg"
output_gcode = f"{base_path}/drawing.ngc"
origin_offset = (-40, -40, 0)


def try_bcnc_internal():
    """Try to use bCNC's internal conversion modules"""
    try:
        # Try different import paths for bCNC modules
        import_attempts = [
            'bCNC.CNC',
            'bCNC.Block', 
            'bCNC.SVG',
            'CNC',
            'Block'
        ]
        
        for module_name in import_attempts:
            try:
                print(f"[INFO] Försöker importera {module_name}...")
                module = __import__(module_name, fromlist=[''])
                print(f"[INFO] Import lyckades: {module_name}")
                print(f"[INFO] Tillgängliga funktioner: {[x for x in dir(module) if not x.startswith('_')]}")
                
                # Look for SVG-related functions
                svg_functions = [x for x in dir(module) if 'svg' in x.lower()]
                if svg_functions:
                    print(f"[INFO] SVG-relaterade funktioner: {svg_functions}")
                
            except ImportError as e:
                print(f"[DEBUG] Kunde inte importera {module_name}: {e}")
                continue
                
    except Exception as e:
        print(f"[FEL] Import misslyckades: {e}")
        return False
    
    return False


def try_bcnc_script_mode():
    """Try to run bCNC in script/batch mode"""
    try:
        import subprocess
        
        # Try to find bCNC installation and run it with script
        script_commands = f"""
load {svg_input}
origin [{origin_offset[0]}] [{origin_offset[1]}] [{origin_offset[2]}]
save {output_gcode}
quit
"""
        
        # Save script to temporary file
        script_file = f"{base_path}/bcnc_script.txt"
        with open(script_file, 'w') as f:
            f.write(script_commands)
        
        # Try to run bCNC with script (if it supports batch mode)
        cmd = ['bcnc', '--script', script_file]
        print(f"[INFO] Försöker köra bCNC i script-läge: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("[INFO] bCNC script-läge lyckades!")
            return True
        else:
            print(f"[DEBUG] Script-läge misslyckades: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("[DEBUG] bCNC script timeout")
    except FileNotFoundError:
        print("[DEBUG] bCNC script-läge inte stödd")
    except Exception as e:
        print(f"[DEBUG] Script-läge fel: {e}")
    
    return False


def try_python_bcnc():
    """Try to run bCNC as Python module"""
    try:
        import subprocess
        
        # Try running bCNC as module with arguments
        cmd = ['python', '-m', 'bCNC', svg_input]
        print(f"[INFO] Försöker köra bCNC som Python modul: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if "error" not in result.stderr.lower():
            print("[INFO] bCNC Python modul körning lyckades!")
            return True
        else:
            print(f"[DEBUG] Python modul misslyckades: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("[DEBUG] Python modul timeout")
    except Exception as e:
        print(f"[DEBUG] Python modul fel: {e}")
    
    return False


def main():
    """Try different approaches to use bCNC headlessly"""
    print("[INFO] Försöker använda bCNC utan GUI...")
    
    approaches = [
        ("Interna moduler", try_bcnc_internal),
        ("Script-läge", try_bcnc_script_mode), 
        ("Python modul", try_python_bcnc)
    ]
    
    for name, func in approaches:
        print(f"\n[INFO] === Försöker {name} ===")
        if func():
            print(f"[INFO] {name} lyckades!")
            return True
        print(f"[DEBUG] {name} misslyckades")
    
    print("\n[FEL] Alla metoder för headless bCNC misslyckades")
    print("[INFO] Rekommendation: Använd standalone_svg_converter.py istället")
    return False


if __name__ == "__main__":
    main()