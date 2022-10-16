set_db / .library designs/example.lib
set CIRCUIT b03
read_hdl ${CIRCUIT}_locked_netlist.v
elaborate ${CIRCUIT}_lbll
ungroup -flatten -all -force
current_design ${CIRCUIT}_lbll

set period 20
create_clock -period $period clk
set_input_delay 0 -clock clk [all_inputs -no_clocks]
set_output_delay 0 -clock clk [all_outputs]
syn_generic
syn_map

set con [open ${CIRCUIT}_LDs.txt]
set lds [split [read $con] "\n"]
close $con;                          # Saves a few bytes :-)
foreach ld $lds {
	set LDlatch [get_db insts $ld]
    echo $LDlatch
	set_case_analysis 0 [get_db $LDlatch .pins -if .base_name==Q]
	
	delete_obj $LDlatch
}

syn_generic
syn_map
write_hdl -generic ${CIRCUIT}_lbll > ${CIRCUIT}_locked_netlist_removed.v

set lats [get_db insts -if .is_latch]
foreach latch $lats {
	set l_name "[get_db $latch .name]"
	report_timing -from $latch > ${CIRCUIT}_time_reports_remove_LD/${l_name}.from
	report_timing -to $latch > ${CIRCUIT}_time_reports_remove_LD/${l_name}.to
}

exit



