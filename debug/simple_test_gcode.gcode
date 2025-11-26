; Test Drawing - Simple Square
G21 ; mm units
G90 ; absolute positioning
G0 Z5 ; lift pen
G0 X0 Y0 ; go to start
G1 Z-1 F500 ; pen down
G1 X10 Y0 F300
G1 X10 Y10 F300
G1 X0 Y10 F300
G1 X0 Y0 F300
G0 Z5 ; pen up
G0 X15 Y0
G1 Z-1 F500
G1 X25 Y0 F300
G1 X25 Y10 F300
G1 X15 Y10 F300
G1 X15 Y0 F300
G0 Z5
G0 X0 Y0 Z10
M30