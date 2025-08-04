#!/usr/bin/env python3
"""
Script to fix the orphaned code block in conscious_cursor_interface_RESTORED.py
"""

def fix_orphaned_code():
    """Remove the orphaned code block from the file."""
    
    # Read the file
    with open('conscious_cursor_interface_RESTORED.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Original file has {len(lines)} lines")
    
    # Find the broken main function at line 2780 (index 2779)
    # Find the good main function at line 2958 (index 2957)
    
    # The problem starts after line 2785: "    interface = CleanCursorInterface()"
    # We need to keep the good main function and remove the orphaned code
    
    new_lines = []
    
    # Copy everything up to and including the interface creation (line 2785)
    for i in range(2785):  # Lines 0-2784
        new_lines.append(lines[i])
    
    # Add the proper try/except/finally block
    new_lines.append("    \n")
    new_lines.append("    try:\n")
    new_lines.append("        # Start the tkinter main loop\n")
    new_lines.append("        interface.root.mainloop()\n")
    new_lines.append("    except KeyboardInterrupt:\n")
    new_lines.append("        print(\"\\n⚠️ Interrupted by user\")\n")
    new_lines.append("    except Exception as e:\n")
    new_lines.append("        print(f\"❌ Error: {e}\")\n")
    new_lines.append("        traceback.print_exc()\n")
    new_lines.append("    finally:\n")
    new_lines.append("        # Cleanup\n")
    new_lines.append("        if hasattr(interface, 'hand_controller') and interface.hand_controller:\n")
    new_lines.append("            try:\n")
    new_lines.append("                interface.hand_controller.cleanup()\n")
    new_lines.append("            except:\n")
    new_lines.append("                pass\n")
    new_lines.append("        print(\"🔌 Clean shutdown complete\")\n")
    new_lines.append("\n")
    new_lines.append("\n")
    new_lines.append("if __name__ == \"__main__\":\n")
    new_lines.append("    main()\n")
    
    print(f"Fixed file will have {len(new_lines)} lines")
    print(f"Removed {len(lines) - len(new_lines)} lines of orphaned code")
    
    # Write the fixed file
    with open('conscious_cursor_interface_RESTORED_FIXED.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print("✅ Fixed file saved as conscious_cursor_interface_RESTORED_FIXED.py")
    return True

if __name__ == "__main__":
    fix_orphaned_code()
