set project [file normalize $::env(MABA_PROJECT)]
open_project $project
open_solution sol1
set limit 3
if {[info exists ::env(MABA_RTL_CASE_LIMIT)]} {set limit $::env(MABA_RTL_CASE_LIMIT)}
set args "[file normalize $::env(MABA_DATA)] $limit"
if {[info exists ::env(MABA_PROFILE)] && $::env(MABA_PROFILE) eq "1"} {
 cosim_design -rtl verilog -trace_level none -enable_dataflow_profiling -argv $args
} else {
 cosim_design -rtl verilog -trace_level none -argv $args
}
exit
