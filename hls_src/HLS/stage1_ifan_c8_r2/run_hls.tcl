set mode "synth"
if {$argc > 0} {
    set mode [lindex $argv 0]
}

open_project stage1_ifan_c8_r2_hls_prj
set_top ifan_stage1_top
add_files ifan_stage1.cpp
add_files ifan_stage1_engines.cpp
add_files -tb test_ifan_stage1.cpp
open_solution "sol1"
set_part {xc7k325tffg900-2}
create_clock -period 5.00 -name default

if {$mode eq "csim"} {
    csim_design
} elseif {$mode eq "synth"} {
    csynth_design
} elseif {$mode eq "cosim"} {
    csim_design
    csynth_design
    cosim_design
} else {
    csim_design
    csynth_design
}

exit
