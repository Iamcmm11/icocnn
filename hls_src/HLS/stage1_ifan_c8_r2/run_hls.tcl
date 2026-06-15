proc env_or_default {name default_value} {
    if {[info exists ::env($name)] && $::env($name) ne ""} {
        return $::env($name)
    }
    return $default_value
}

set project_name [env_or_default "HLS_PROJECT" "stage1_ifan_c8_r2_frontend_hls_prj"]
set solution_name [env_or_default "HLS_SOLUTION" "sol1"]
set top_name [env_or_default "HLS_TOP" "ifan_dual_frontend_top"]
set part_name [env_or_default "HLS_PART" "xc7k325tffg900-2"]
set clock_period [env_or_default "HLS_CLOCK" "5.0"]
set default_project_root [file normalize [file join [pwd] "_hls_work"]]
set project_root [env_or_default "ICO_HLS_PROJECT_ROOT" $default_project_root]
set source_dir [file normalize [env_or_default "ICO_HLS_SOURCE_DIR" [pwd]]]
set project_name [env_or_default "ICO_HLS_PROJECT" $project_name]
set solution_name [env_or_default "ICO_HLS_SOLUTION" $solution_name]
set top_name [env_or_default "ICO_HLS_TOP" $top_name]
set part_name [env_or_default "ICO_HLS_PART" $part_name]
set clock_period [env_or_default "ICO_HLS_CLOCK" $clock_period]
set mode [string tolower [env_or_default "ICO_HLS_MODE" "synth"]]
set module_name [string tolower [env_or_default "ICO_HLS_MODULE" "frontend"]]
set cppflags [env_or_default "ICO_HLS_CPPFLAGS" ""]

puts "=== Vitis HLS Terminal Flow ==="
puts "Project  : $project_name"
puts "ProjRoot : $project_root"
puts "SrcDir   : $source_dir"
puts "Solution : $solution_name"
puts "Top      : $top_name"
puts "Part     : $part_name"
puts "Clock(ns): $clock_period"
puts "Mode     : $mode"
puts "Module   : $module_name"
puts "CppFlags : $cppflags"

set project_path [file normalize [file join $project_root $project_name]]
file mkdir $project_root
open_project -reset $project_path
set_top $top_name
set saved_pwd [pwd]
cd $source_dir
if {$module_name eq "temporal"} {
    set temporal_stage_dir [file normalize [file join $project_root temporal_r1]]
    set legacy_stage_dir [file normalize [file join $project_root full_stage1_legacy]]
    file mkdir $temporal_stage_dir
    file mkdir $legacy_stage_dir
    file copy -force [file join $source_dir temporal_r1 ifan_temporal_r1.cpp] [file join $temporal_stage_dir ifan_temporal_r1.cpp]
    file copy -force [file join $source_dir temporal_r1 ifan_temporal_r1.hpp] [file join $temporal_stage_dir ifan_temporal_r1.hpp]
    file copy -force [file join $source_dir temporal_r1 test_ifan_temporal_r1.cpp] [file join $temporal_stage_dir test_ifan_temporal_r1.cpp]
    file copy -force [file join $source_dir full_stage1_legacy ifan_stage1.hpp] [file join $legacy_stage_dir ifan_stage1.hpp]
    if {$cppflags eq ""} {
        add_files [file join $temporal_stage_dir ifan_temporal_r1.cpp]
        add_files [file join $temporal_stage_dir ifan_temporal_r1.hpp]
        add_files [file join $legacy_stage_dir ifan_stage1.hpp]
        add_files -tb [file join $temporal_stage_dir test_ifan_temporal_r1.cpp]
    } else {
        add_files -cflags $cppflags [file join $temporal_stage_dir ifan_temporal_r1.cpp]
        add_files [file join $temporal_stage_dir ifan_temporal_r1.hpp]
        add_files [file join $legacy_stage_dir ifan_stage1.hpp]
        add_files -tb -cflags $cppflags [file join $temporal_stage_dir test_ifan_temporal_r1.cpp]
    }
} else {
    if {$cppflags eq ""} {
        add_files frontend_dual_feature/ifan_dual_frontend.cpp
        add_files full_stage1_legacy/ifan_stage1_engines.cpp
        add_files -tb frontend_dual_feature/test_ifan_dual_frontend.cpp
    } else {
        add_files -cflags $cppflags frontend_dual_feature/ifan_dual_frontend.cpp
        add_files -cflags $cppflags full_stage1_legacy/ifan_stage1_engines.cpp
        add_files -tb -cflags $cppflags frontend_dual_feature/test_ifan_dual_frontend.cpp
    }
}
cd $saved_pwd
open_solution -reset $solution_name
set_part $part_name
create_clock -period $clock_period -name default

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
