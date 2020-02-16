reset
set nokey
set xrange [0:1]
set yrange [0:1]
set size square

set datafile separator whitespace
set term gif animate
set output "fsample.gif"

n0 = 1
n1 = 99
dn = 1

load "pltscript.plt"
