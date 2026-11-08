proc opt {key fallback} {if {[info exists ::env($key)]} {return $::env($key)};return $fallback}
set src [file dirname [file normalize [info script]]]
set variant [opt MABA_VARIANT 3]
set tile [opt MABA_TILE 16]
set par [opt MABA_PAR 1]
set mode [opt MABA_MODE all]
set fifo [opt MABA_FIFO 8]
set frames [opt MABA_T 6]
set positions [opt MABA_POSITIONS 240]
set project [opt MABA_PROJECT [file join $src _work v${variant}_t${tile}_p${par}]]
set data [opt MABA_DATA [file join $src artifacts data]]
set flags "-std=c++14 -DMABA_VARIANT=$variant -DMABA_TILE=$tile -DMABA_PAR=$par -DMABA_FIFO=$fifo -DMABA_T=$frames -DMABA_POSITIONS=$positions"
set project [file normalize $project]
set staged [file join [file dirname $project] sources_[file tail $project]]
file mkdir $staged
foreach f {maba_top.cpp maba_top.hpp test_maba.cpp} {file copy -force [file join $src $f] [file join $staged $f]}
cd [file dirname $project]
open_project -reset $project
set_top maba_top
add_files [file join $staged maba_top.cpp] -cflags $flags
add_files -tb [file join $staged test_maba.cpp] -cflags $flags
open_solution -reset sol1
set_part xc7vx485tffg1761-2
create_clock -period 5.0 -name default
# Keep default safe floating-point reassociation (do not enable unsafe math).
set argv "[file normalize $data] [opt MABA_CASE_LIMIT 0]"
csim_design -O -argv $argv
if {$mode ne "csim"} {csynth_design}
if {$mode eq "all" || $mode eq "cosim"} {
 if {[opt MABA_PROFILE 0] eq "1"} {
  cosim_design -rtl verilog -trace_level none -enable_dataflow_profiling -argv "[file normalize $data] [opt MABA_RTL_CASE_LIMIT 3]"
 } else {
  cosim_design -rtl verilog -trace_level none -argv "[file normalize $data] [opt MABA_RTL_CASE_LIMIT 3]"
 }
}
exit
