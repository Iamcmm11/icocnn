proc env_or_default {name default_value} {
    if {[info exists ::env($name)] && $::env($name) ne ""} {
        return $::env($name)
    }
    return $default_value
}

set project_name [env_or_default "HLS_PROJECT" "layer2_5_hls_prj"]
set solution_name [env_or_default "HLS_SOLUTION" "sol1"]
set top_name [env_or_default "HLS_TOP" "conv_ico_layer2_5"]
set part_name [env_or_default "HLS_PART" "xc7k325tffg900-2"]
set clock_period [env_or_default "HLS_CLOCK" "5.0"]
set project_name [env_or_default "ICO_HLS_PROJECT" $project_name]
set solution_name [env_or_default "ICO_HLS_SOLUTION" $solution_name]
set top_name [env_or_default "ICO_HLS_TOP" $top_name]
set part_name [env_or_default "ICO_HLS_PART" $part_name]
set clock_period [env_or_default "ICO_HLS_CLOCK" $clock_period]
set mode [string tolower [env_or_default "ICO_HLS_MODE" "quick"]]

puts "=== Vitis HLS Terminal Flow ==="
puts "Project  : $project_name"
puts "Solution : $solution_name"
puts "Top      : $top_name"
puts "Part     : $part_name"
puts "Clock(ns): $clock_period"
puts "Mode     : $mode"

if {[file exists $project_name]} {
    puts "Removing existing project: $project_name"
    file delete -force $project_name
}

open_project $project_name
set_top $top_name

add_files ico_conv_layer2_5.cpp
add_files ico_conv_layer2_5.hpp
add_files ../common/utils.hpp
add_files -tb test_ico_conv_layer2_5.cpp

open_solution -reset $solution_name
set_part $part_name
create_clock -period $clock_period -name default

if {$mode eq "csim"} {
    csim_design -clean
} elseif {$mode eq "synth"} {
    csynth_design
} elseif {$mode eq "cosim"} {
    csynth_design
    cosim_design
} elseif {$mode eq "export"} {
    csynth_design
    export_design -format ip_catalog
} elseif {$mode eq "all"} {
    csim_design -clean
    csynth_design
    cosim_design
    export_design -format ip_catalog
} else {
    csim_design -clean
    csynth_design
}

exit
