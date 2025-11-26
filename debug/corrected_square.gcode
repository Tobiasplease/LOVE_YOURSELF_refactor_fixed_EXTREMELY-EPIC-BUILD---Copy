G21 ; Set units to millimeters
G90 ; Use absolute positioning
G17 ; Select XY plane

G0 X48.6807 Y15.5740 ; Move to start
M3 S50 ; Lower pen
G1 X46.4556 Y17.8319 F1500 ; Draw line
G1 X35.8966 Y28.5461 F1500 ; Draw line
G1 X37.9207 Y26.4922 F1500 ; Draw line
G1 X48.6807 Y15.5740 F1500 ; Draw line

M3 S30 ; Raise pen
G0 X0 Y0 ; Return to origin
M2 ; End program