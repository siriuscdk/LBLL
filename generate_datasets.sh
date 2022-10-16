#!/bin/bash
seed=1
# update all synthesized scripts
#python3 update_syn_scripts.py $seed
# remove all timing reports to avoid duplicate
#rm -r ./syn_s298/s298_time_reports
rm -r ./syn_s298/s298_time_reports_remove_LD
#rm -r ./syn_s9234/s9234_time_reports
rm -r ./syn_s9234/s9234_time_reports_remove_LD
#rm -r ./syn_s13207/s13207_time_reports
rm -r ./syn_s13207/s13207_time_reports_remove_LD
#rm -r ./syn_s15850/s15850_time_reports
rm -r ./syn_s15850/s15850_time_reports_remove_LD
#rm -r ./syn_s35932/s35932_time_reports
rm -r ./syn_s35932/s35932_time_reports_remove_LD
#rm -r ./syn_s38417/s38417_time_reports
rm -r ./syn_s38417/s38417_time_reports_remove_LD
#rm -r ./syn_s38584/s38584_time_reports
rm -r ./syn_s38584/s38584_time_reports_remove_LD
#rm -r ./syn_b03/b03_time_reports
rm -r ./syn_b03/b03_time_reports_remove_LD
#rm -r ./syn_b04/b04_time_reports
rm -r ./syn_b04/b04_time_reports_remove_LD
#rm -r ./syn_b07/b07_time_reports
rm -r ./syn_b07/b07_time_reports_remove_LD
#rm -r ./syn_b11/b11_time_reports
rm -r ./syn_b11/b11_time_reports_remove_LD
#rm -r ./syn_b12/b12_time_reports
rm -r ./syn_b12/b12_time_reports_remove_LD
#rm -r ./syn_b13/b13_time_reports
rm -r ./syn_b13/b13_time_reports_remove_LD
#rm -r ./syn_b14/b14_time_reports
rm -r ./syn_b14/b14_time_reports_remove_LD
#rm -r ./syn_b15/b15_time_reports
rm -r ./syn_b15/b15_time_reports_remove_LD
#rm -r ./syn_b17/b17_time_reports
rm -r ./syn_b17/b17_time_reports_remove_LD
#rm -r ./syn_b20/b20_time_reports
rm -r ./syn_b20/b20_time_reports_remove_LD
#rm -r ./syn_b21/b21_time_reports
rm -r ./syn_b21/b21_time_reports_remove_LD
#rm -r ./syn_b22/b22_time_reports
rm -r ./syn_b22/b22_time_reports_remove_LD

# synthesize and generate new netlist
cd ./syn_s298
#genus -files syn_rename_s298.tcl
genus -files remove_LD_s298.tcl
cd ../syn_s9234
#genus -files syn_rename_s9234.tcl
genus -files remove_LD_s9234.tcl
cd ../syn_s13207
#genus -files syn_rename_s13207.tcl
genus -files remove_LD_s13207.tcl
cd ../syn_s15850
#genus -files syn_rename_s15850.tcl
genus -files remove_LD_s15850.tcl
cd ../syn_s35932
#genus -files syn_rename_s35932.tcl
genus -files remove_LD_s35932.tcl
cd ../syn_s38417
#genus -files syn_rename_s38417.tcl
genus -files remove_LD_s38417.tcl
cd ../syn_s38584
#genus -files syn_rename_s38584.tcl
genus -files remove_LD_s38584.tcl
cd ../syn_b03
#genus -files syn_rename_b03.tcl
genus -files remove_LD_b03.tcl
cd ../syn_b04
#genus -files syn_rename_b04.tcl
genus -files remove_LD_b04.tcl
cd ../syn_b07
#genus -files syn_rename_b07.tcl
genus -files remove_LD_b07.tcl
cd ../syn_b11
#genus -files syn_rename_b11.tcl
genus -files remove_LD_b11.tcl
cd ../syn_b12
#genus -files syn_rename_b12.tcl
genus -files remove_LD_b12.tcl
cd ../syn_b13
#genus -files syn_rename_b13.tcl
genus -files remove_LD_b13.tcl
cd ../syn_b14
#genus -files syn_rename_b14.tcl
genus -files remove_LD_b14.tcl
cd ../syn_b15
#genus -files syn_rename_b15.tcl
genus -files remove_LD_b15.tcl
cd ../syn_b17
#genus -files syn_rename_b17.tcl
genus -files remove_LD_b17.tcl
cd ../syn_b20
#genus -files syn_rename_b20.tcl
genus -files remove_LD_b20.tcl
cd ../syn_b21
#genus -files syn_rename_b21.tcl
genus -files remove_LD_b21.tcl
cd ../syn_b22
#genus -files syn_rename_b22.tcl
genus -files remove_LD_b22.tcl

cd ..
mkdir -p conversion_remove_LD
cp ./v2bench_translate_cmu_delay_remove_LD.py ./conversion_remove_LD

# copy netlists
cp ./syn_s298/s298_locked_netlist_removed.v ./conversion_remove_LD
cp ./syn_s9234/s9234_locked_netlist_removed.v ./conversion_remove_LD
cp ./syn_s13207/s13207_locked_netlist_removed.v ./conversion_remove_LD
cp ./syn_s15850/s15850_locked_netlist_removed.v ./conversion_remove_LD
cp ./syn_s35932/s35932_locked_netlist_removed.v ./conversion_remove_LD
cp ./syn_s38417/s38417_locked_netlist_removed.v ./conversion_remove_LD
cp ./syn_s38584/s38584_locked_netlist_removed.v ./conversion_remove_LD
cp ./syn_b03/b03_locked_netlist_removed.v ./conversion_remove_LD
cp ./syn_b04/b04_locked_netlist_removed.v ./conversion_remove_LD
cp ./syn_b07/b07_locked_netlist_removed.v ./conversion_remove_LD
cp ./syn_b11/b11_locked_netlist_removed.v ./conversion_remove_LD
cp ./syn_b12/b12_locked_netlist_removed.v ./conversion_remove_LD
cp ./syn_b13/b13_locked_netlist_removed.v ./conversion_remove_LD
cp ./syn_b14/b14_locked_netlist_removed.v ./conversion_remove_LD
cp ./syn_b15/b15_locked_netlist_removed.v ./conversion_remove_LD
cp ./syn_b17/b17_locked_netlist_removed.v ./conversion_remove_LD
cp ./syn_b20/b20_locked_netlist_removed.v ./conversion_remove_LD
cp ./syn_b21/b21_locked_netlist_removed.v ./conversion_remove_LD
cp ./syn_b22/b22_locked_netlist_removed.v ./conversion_remove_LD

# convert verilog netlist to bench 
cd conversion_remove_LD
python3 v2bench_translate_cmu_delay_remove_LD.py
cd ..

# get bench and timing reports and sort them out
mkdir -p "remove_LD/s298_$seed"
cp ./conversion_remove_LD/s298_clean_remove_LD.bench ./"remove_LD/s298_"$seed
cp ./conversion_remove_LD/s298_latchname2Q_remove_LD ./"remove_LD/s298_"$seed
cp -r ./syn_s298/s298_time_reports_remove_LD ./"remove_LD/s298_"$seed

mkdir -p "remove_LD/s9234_$seed"
cp ./conversion_remove_LD/s9234_clean_remove_LD.bench ./"remove_LD/s9234_"$seed
cp ./conversion_remove_LD/s9234_latchname2Q_remove_LD ./"remove_LD/s9234_"$seed
cp -r ./syn_s9234/s9234_time_reports_remove_LD ./"remove_LD/s9234_"$seed

mkdir -p "remove_LD/s13207_"$seed
cp ./conversion_remove_LD/s13207_clean_remove_LD.bench ./"remove_LD/s13207_"$seed
cp ./conversion_remove_LD/s13207_latchname2Q_remove_LD ./"remove_LD/s13207_"$seed
cp -r ./syn_s13207/s13207_time_reports_remove_LD ./"remove_LD/s13207_"$seed

mkdir -p "remove_LD/s15850_"$seed
cp ./conversion_remove_LD/s15850_clean_remove_LD.bench ./"remove_LD/s15850_"$seed
cp ./conversion_remove_LD/s15850_latchname2Q_remove_LD ./"remove_LD/s15850_"$seed
cp -r ./syn_s15850/s15850_time_reports_remove_LD ./"remove_LD/s15850_"$seed

mkdir -p "remove_LD/s35932_"$seed
cp ./conversion_remove_LD/s35932_clean_remove_LD.bench ./"remove_LD/s35932_"$seed
cp ./conversion_remove_LD/s35932_latchname2Q_remove_LD ./"remove_LD/s35932_"$seed
cp -r ./syn_s35932/s35932_time_reports_remove_LD ./"remove_LD/s35932_"$seed

mkdir -p "remove_LD/s38417_"$seed
cp ./conversion_remove_LD/s38417_clean_remove_LD.bench ./"remove_LD/s38417_"$seed
cp ./conversion_remove_LD/s38417_latchname2Q_remove_LD ./"remove_LD/s38417_"$seed
cp -r ./syn_s38417/s38417_time_reports_remove_LD ./"remove_LD/s38417_"$seed

mkdir -p "remove_LD/s38584_"$seed
cp ./conversion_remove_LD/s38584_clean_remove_LD.bench ./"remove_LD/s38584_"$seed
cp ./conversion_remove_LD/s38584_latchname2Q_remove_LD ./"remove_LD/s38584_"$seed
cp -r ./syn_s38584/s38584_time_reports_remove_LD ./"remove_LD/s38584_"$seed

mkdir -p "remove_LD/b03_"$seed
cp ./conversion_remove_LD/b03_clean_remove_LD.bench ./"remove_LD/b03_"$seed
cp ./conversion_remove_LD/b03_latchname2Q_remove_LD ./"remove_LD/b03_"$seed
cp -r ./syn_b03/b03_time_reports_remove_LD ./"remove_LD/b03_"$seed

mkdir -p "remove_LD/b04_"$seed
cp ./conversion_remove_LD/b04_clean_remove_LD.bench ./"remove_LD/b04_"$seed
cp ./conversion_remove_LD/b04_latchname2Q_remove_LD ./"remove_LD/b04_"$seed
cp -r ./syn_b04/b04_time_reports_remove_LD ./"remove_LD/b04_"$seed

mkdir -p "remove_LD/b07_"$seed
cp ./conversion_remove_LD/b07_clean_remove_LD.bench ./"remove_LD/b07_"$seed
cp ./conversion_remove_LD/b07_latchname2Q_remove_LD ./"remove_LD/b07_"$seed
cp -r ./syn_b07/b07_time_reports_remove_LD ./"remove_LD/b07_"$seed

mkdir -p "remove_LD/b11_"$seed
cp ./conversion_remove_LD/b11_clean_remove_LD.bench ./"remove_LD/b11_"$seed
cp ./conversion_remove_LD/b11_latchname2Q_remove_LD ./"remove_LD/b11_"$seed
cp -r ./syn_b11/b11_time_reports_remove_LD ./"remove_LD/b11_"$seed

mkdir -p "remove_LD/b12_"$seed
cp ./conversion_remove_LD/b12_clean_remove_LD.bench ./"remove_LD/b12_"$seed
cp ./conversion_remove_LD/b12_latchname2Q_remove_LD ./"remove_LD/b12_"$seed
cp -r ./syn_b12/b12_time_reports_remove_LD ./"remove_LD/b12_"$seed

mkdir -p "remove_LD/b13_"$seed
cp ./conversion_remove_LD/b13_clean_remove_LD.bench ./"remove_LD/b13_"$seed
cp ./conversion_remove_LD/b13_latchname2Q_remove_LD ./"remove_LD/b13_"$seed
cp -r ./syn_b13/b13_time_reports_remove_LD ./"remove_LD/b13_"$seed

mkdir -p "remove_LD/b14_"$seed
cp ./conversion_remove_LD/b14_clean_remove_LD.bench ./"remove_LD/b14_"$seed
cp ./conversion_remove_LD/b14_latchname2Q_remove_LD ./"remove_LD/b14_"$seed
cp -r ./syn_b14/b14_time_reports_remove_LD ./"remove_LD/b14_"$seed

mkdir -p "remove_LD/b15_"$seed
cp ./conversion_remove_LD/b15_clean_remove_LD.bench ./"remove_LD/b15_"$seed
cp ./conversion_remove_LD/b15_latchname2Q_remove_LD ./"remove_LD/b15_"$seed
cp -r ./syn_b15/b15_time_reports_remove_LD ./"remove_LD/b15_"$seed

mkdir -p "remove_LD/b17_"$seed
cp ./conversion_remove_LD/b17_clean_remove_LD.bench ./"remove_LD/b17_"$seed
cp ./conversion_remove_LD/b17_latchname2Q_remove_LD ./"remove_LD/b17_"$seed
cp -r ./syn_b17/b17_time_reports_remove_LD ./"remove_LD/b17_"$seed

mkdir -p "remove_LD/b20_"$seed
cp ./conversion_remove_LD/b20_clean_remove_LD.bench ./"remove_LD/b20_"$seed
cp ./conversion_remove_LD/b20_latchname2Q_remove_LD ./"remove_LD/b20_"$seed
cp -r ./syn_b20/b20_time_reports_remove_LD ./"remove_LD/b20_"$seed

mkdir -p "remove_LD/b21_"$seed
cp ./conversion_remove_LD/b21_clean_remove_LD.bench ./"remove_LD/b21_"$seed
cp ./conversion_remove_LD/b21_latchname2Q_remove_LD ./"remove_LD/b21_"$seed
cp -r ./syn_b21/b21_time_reports_remove_LD ./"remove_LD/b21_"$seed

mkdir -p "remove_LD/b22_"$seed
cp ./conversion_remove_LD/b22_clean_remove_LD.bench ./"remove_LD/b22_"$seed
cp ./conversion_remove_LD/b22_latchname2Q_remove_LD ./"remove_LD/b22_"$seed
cp -r ./syn_b22/b22_time_reports_remove_LD ./"remove_LD/b22_"$seed



