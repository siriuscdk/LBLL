import sys
import os
import re
import glob

def translate(latchnamePath, filePath):
    f = open(filePath, "r")
    benchFilePath = filePath.replace(".v", ".bench")

    of = open(benchFilePath, "w+")
    parsed_f = (f.read()).split(";")

    latchname2Q={}
    flipflop_D_ports = []
    flipflop_Q_ports = []
    flipflop_QN_ports = []

    
    INPUT_BUS_REGEX = r"^(\s*)input\s+(\s*\[[0-9:]+\][\w\d\s\\\/,]+)+;$"
    OUTPUT_BUS_REGEX = r"^(\s*)output\s+(\s*\[[0-9:]+\][\w\d\s\\\/,]+)+;$"
    BUS_NAME_REGEX = r"\[[0-9:]+\][\w\d\s\\\/]+[,;]" 
    INPUT_REGEX = r"^(\s*)input\s+([\w\d\s\\\/\[\]:,]+)+;$"
    INPUT_NAME_REGEX = r"[\w\d\\\/\[\]]+[,;]"
    OUTPUT_REGEX = r"^(\s*)output\s+([\w\d\s\\\/\[\],]+)+;$"
    OUTPUT_NAME_REGEX = r"[\w\d\\\/\[\]]+[,;]"
    ASSIGN_REGEX = r"^(\s*)assign\s+[\w\d\\\/\[\]]+\s+=\s+[\w\d\'\\\/\[\]]+;$"
    DFF_REGEX1 = r"^(\s*)fflopd[\w\d\s\\\/\[\]]*\([\w\d\s,\(\)\\\/\[\]\'\.]+\);$"
    DFF_REGEX2 = r"^(\s*)flopdrs[\w\d\s\\\/\[\]]*\([\w\d\s,\(\)\\\/\[\]\'\.]+\);$"
    DIN_REGEX = r"\.D[\s]*\([\w\d\s\\\/\[\]]+\)"
    Q_REGEX = r"\.Q[\s]*\([\w\d\s\\\/\[\]]+\)"
    QN_REGEX = r"\.QN[\s]*\([\w\d\s\\\/\[\]]+\)"
    # match latch
    LATCH_REGEX = r"^(\s*)latchdr[\w\d\s\\\/\[\]]*\([\w\d\s,\(\)\\\/\[\]\'\.]+\);$"
    LATCH_REGEX2="^\s*latchdr\s*([\w\d\s\\\/\[\]]+)\([\w\d\s,\(\)\\\/\[\]\'\.]+\);$"
    LATCH_DIN_REGEX = r"\.D[\s]*\([\w\d\s\\\/\[\]]+\)"
    LATCH_Q_REGEX = r"\.Q[\s]*\([\w\d\s\\\/\[\]]+\)"
    LATCH_QN_REGEX = r"\.QN[\s]*\([\w\d\s\\\/\[\]]+\)"

    # all gates, the first pin is output
    GATE_REGEX = r"^\s*[\w\d]+\s+[\w\d\\\/\[\]]+\s+\([\w\d\s,\(\)\\\/\[\]\.]+\);$"
    GATE_TYPE_NAME_REGEX = r"^\s*[\w\d]+\s+[\w\d\\\/\[\]]+"

    DIN_INV_REGEX = r"\s*[\w\d\s\\\/\[\]]+\)"
    Z_REGEX = r"\s*\([\w\d\s\\\/\[\]]+,"

    DIN1_REGEX = r",\s*[\w\d\s\\\/\[\]]+,\s*"
    DIN2_REGEX = r"\s*[\w\d\s\\\/\[\]]+\)"





    for i in range(len(parsed_f)):
        parsed_f[i] += ";"
        parsed_f[i] = parsed_f[i].replace("\n", " ")


        #------------------------------------
        # Match Bus Input (example: input [9:0] address;)
        #------------------------------------
        if re.match(INPUT_BUS_REGEX, parsed_f[i]):
            x = re.findall(BUS_NAME_REGEX, parsed_f[i])
            for j in range(len(x)):
                y = re.split(r"\[|:|\]| |,|;", x[j])
                for k in range(int(y[2]), int(y[1])+1):
                    of.write("INPUT("+y[-2]+"["+str(k)+"])\n")
                    #input_list.append(str(y[-2]) + "[" + str(k) + "]")
                    #input_visit.append(0)
            continue

        #------------------------------------
        # Match Bus Output (example: input [9:0] address;)
        #------------------------------------
        if re.match(OUTPUT_BUS_REGEX, parsed_f[i]):
            x = re.findall(BUS_NAME_REGEX, parsed_f[i])
            for j in range(len(x)):
                y = re.split(r"\[|:|\]| |,|;", x[j])
                for k in range(int(y[2]), int(y[1])+1):
                    of.write("OUTPUT("+y[-2]+"["+str(k)+"])\n")

            continue

        #------------------------------------
        # Match Input (example: input clk, rst;)
        #------------------------------------
        if re.match(INPUT_REGEX, parsed_f[i]): 
            x = re.findall(INPUT_NAME_REGEX, parsed_f[i])
            for j in range(len(x)):
                y = re.sub(r",|;", "", x[j])

                if (y == "VDD") or (y == "GND"):
                    continue
                of.write("INPUT("+y+")\n")

            continue

        #------------------------------------
        # Match Output
        #------------------------------------
        if re.match(OUTPUT_REGEX, parsed_f[i]): 
            x = re.findall(OUTPUT_NAME_REGEX, parsed_f[i])
            for j in range(len(x)):
                y = re.sub(r",|;", "", x[j])
                of.write("OUTPUT("+y+")\n")
            continue
        
        #------------------------------------
        # Match Assign (example: assign x = 1'b1;)
        #------------------------------------
        if re.match(ASSIGN_REGEX, parsed_f[i]): 
            x = re.split(r"\s+|=|;", parsed_f[i])
            if x[-2] == "1'b1":
                of.write(x[2]+" = XNOR(reset, reset)\n")
            elif x[-2] == "1'b0":
                of.write(x[2]+" = XOR(reset, reset)\n")
            elif x[-2] == "clock" or x[-2] == "clk" or x[-2] == "CLOCK" or x[-2] == "CLK":
                pass
            else:
                of.write(x[2]+" = BUFF("+x[-2]+")\n")
            continue

        #------------------------------------
        # Match DFF1
        #------------------------------------
        if re.match(DFF_REGEX1, parsed_f[i]):
            #------------------------------------
            # Match D Port
            #------------------------------------
            D = re.search(DIN_REGEX, parsed_f[i])
            if (D):
                D_port = re.split(r"\(|\)", D.group())[1]
            else:
                exit("DIN port doesn't exist!")
            #------------------------------------
            # Match Q Port
            #------------------------------------
            Q = re.search(Q_REGEX, parsed_f[i])
            Q_port = ""
            if (Q):
                Q_port = re.split(r"\(|\)", Q.group())[1].replace(" ", "")
                if (D_port in flipflop_D_ports):
                    DFF_index = flipflop_D_ports.index(D_port)
                    if (flipflop_Q_ports[DFF_index] != ""):
                        of.write(Q_port+" = BUFF("+flipflop_Q_ports[DFF_index]+")\n")
                    elif (flipflop_QN_ports[DFF_index] != ""):
                        of.write(Q_port+" = NOT("+flipflop_QN_ports[DFF_index]+")\n")
                else:
                    of.write(Q_port+" = DFF("+D_port+")\n")
            #------------------------------------
            # Match QN Port
            #------------------------------------
            QN = re.search(QN_REGEX, parsed_f[i])
            QN_port = ""
            if (QN):
                QN_port = re.split(r"\(|\)", QN.group())[1].replace(" ", "")
                if (Q):
                    of.write(QN_port+" = NOT("+Q_port+")\n")
                else:
                    if (D_port in flipflop_D_ports):
                        DFF_index = flipflop_D_ports.index(D_port)
                        if (flipflop_QN_ports[DFF_index] != ""):
                            of.write(QN_port+" = BUFF("+flipflop_QN_ports[DFF_index]+")\n")
                        elif (flipflop_Q_ports[DFF_index] != ""):
                            of.write(QN_port+" = NOT("+flipflop_Q_ports[DFF_index]+")\n")
                    else:
                        of.write(QN_port+"_bar = DFF("+D_port+")\n")
                        of.write(QN_port+" = NOT("+QN_port+"_bar)\n")
            flipflop_D_ports.append(D_port)
            flipflop_Q_ports.append(Q_port)
            flipflop_QN_ports.append(QN_port)
            continue


        #------------------------------------
        # Match DFF2
        #------------------------------------
        if re.match(DFF_REGEX2, parsed_f[i]):
            #------------------------------------
            # Match D Port
            #------------------------------------
            D = re.search(DIN_REGEX, parsed_f[i])
            if (D):
                D_port = re.split(r"\(|\)", D.group())[1]
            else:
                exit("DIN port doesn't exist!")
            #------------------------------------
            # Match Q Port
            #------------------------------------
            Q = re.search(Q_REGEX, parsed_f[i])
            Q_port = ""
            if (Q):
                Q_port = re.split(r"\(|\)", Q.group())[1].replace(" ", "")
                if (D_port in flipflop_D_ports):
                    DFF_index = flipflop_D_ports.index(D_port)
                    if (flipflop_Q_ports[DFF_index] != ""):
                        of.write(Q_port+" = BUFF("+flipflop_Q_ports[DFF_index]+")\n")
                    elif (flipflop_QN_ports[DFF_index] != ""):
                        of.write(Q_port+" = NOT("+flipflop_QN_ports[DFF_index]+")\n")
                else:
                    of.write(Q_port+" = DFF("+D_port+")\n")
            #------------------------------------
            # Match QN Port
            #------------------------------------
            QN = re.search(QN_REGEX, parsed_f[i])
            QN_port = ""
            if (QN):
                QN_port = re.split(r"\(|\)", QN.group())[1].replace(" ", "")
                if (Q):
                    of.write(QN_port+" = NOT("+Q_port+")\n")
                else:
                    if (D_port in flipflop_D_ports):
                        DFF_index = flipflop_D_ports.index(D_port)
                        if (flipflop_QN_ports[DFF_index] != ""):
                            of.write(QN_port+" = BUFF("+flipflop_QN_ports[DFF_index]+")\n")
                        elif (flipflop_Q_ports[DFF_index] != ""):
                            of.write(QN_port+" = NOT("+flipflop_Q_ports[DFF_index]+")\n")
                    else:
                        of.write(QN_port+"_bar = DFF("+D_port+")\n")
                        of.write(QN_port+" = NOT("+QN_port+"_bar)\n")
            flipflop_D_ports.append(D_port)
            flipflop_Q_ports.append(Q_port)
            flipflop_QN_ports.append(QN_port)
            continue

        #------------------------------------
        # Match Latch
        #------------------------------------
        if re.match(LATCH_REGEX, parsed_f[i]):
            x = re.findall(LATCH_REGEX2, parsed_f[i])
            #print (x)
            latch_name=x[0]
            #------------------------------------
            # Match D Port
            #------------------------------------
            D = re.search(LATCH_DIN_REGEX, parsed_f[i])
            if (D):
                D_port = re.split(r"\(|\)", D.group())[1]

            else:
                exit("DIN port doesn't exist!")
            #------------------------------------
            # Match Q Port
            #------------------------------------
            Q = re.search(LATCH_Q_REGEX, parsed_f[i])
            Q_port = ""
            if (Q):
                Q_port = re.split(r"\(|\)", Q.group())[1].replace(" ", "")
                #print (parsed_f[i])
                if "_L0" in latch_name:
                    latch_type="L0"
                elif "_L1" in latch_name:
                    latch_type = "L1"
                elif "_LD" in latch_name:
                    latch_type = "LD"
                elif "_DD" in latch_name:
                    latch_type = "DD"
                of.write(Q_port+f" = LATCH_{latch_type}("+D_port+")\n")
                latchname2Q[Q_port] = latch_name.rstrip() + "_qi_reg"
            #------------------------------------
            # Match QN Port
            #------------------------------------
            QN = re.search(LATCH_QN_REGEX, parsed_f[i])
            QN_port = ""
            if (QN):
                QN_port = re.split(r"\(|\)", QN.group())[1].replace(" ", "")
                if (Q):
                    of.write(QN_port+" = NOT("+Q_port+")\n")
                else:
                    if (D_port in flipflop_D_ports):
                        DFF_index = flipflop_D_ports.index(D_port)
                        print ("double FF")
                        if (flipflop_QN_ports[DFF_index] != ""):
                            of.write(QN_port+" = BUFF("+flipflop_QN_ports[DFF_index]+")\n")
                        elif (flipflop_Q_ports[DFF_index] != ""):
                            of.write(QN_port+" = NOT("+flipflop_Q_ports[DFF_index]+")\n")
                    else:
                        of.write(QN_port+"_bar = DFF("+D_port+")\n")
                        of.write(QN_port+" = NOT("+QN_port+"_bar)\n")
            flipflop_D_ports.append(D_port)
            flipflop_Q_ports.append(Q_port)
            flipflop_QN_ports.append(QN_port)
            continue

        #------------------------------------
        # Match Gates (NAND, AND, NOR, OR, XOR, XNOR, INV, BUF)
        #------------------------------------
        if re.match(GATE_REGEX, parsed_f[i]):
            gate = re.match(GATE_TYPE_NAME_REGEX, parsed_f[i])
            gate_type = re.split(r"\s+", gate.group())[1]
            #------------------------------------
            # For INV (and BUF)
            #------------------------------------
            #if ("hi" in gate_type) or ("ib" in gate_type) or ("nb" in gate_type):
            if ("not" in gate_type):
                #------------------------------------
                # Match DIN Port
                #------------------------------------
                DIN = re.search(DIN_INV_REGEX, parsed_f[i])
                #print (DIN)

                if (DIN):
                    DIN_port = re.split(r"\)", DIN.group())[0].replace(" ", "")

                else:
                    exit("DIN port doesn't exist!")
            
                #------------------------------------
                # Match Z/ZN Port
                #------------------------------------
                Z = re.search(Z_REGEX, parsed_f[i])


                if (Z):
                    Z_port = re.split(r"\(|,", Z.group())[1].replace(" ", "")
                    if "not" in gate_type:
                        of.write(Z_port+" = NOT("+DIN_port+")\n")


                else:
                    exit("Z/ZN port doesn't exist!")
            
            #------------------------------------
            # For NAND, AND, NOR, OR, XOR, XNOR
            #------------------------------------
            else:
                #------------------------------------
                # Match DIN1 Port
                #------------------------------------
                DIN1 = re.search(DIN1_REGEX, parsed_f[i])

                if (DIN1):
                    DIN1_port = re.split(r",", DIN1.group())[1].replace(" ", "")
                    #if (DIN1_port) in input_list:
                        #input_visit[input_list.index(DIN1_port)] = 1
                    #print (DIN1_port)

                else:
                    #print (parsed_f[i])
                    exit("DIN1 port doesn't exist!")
                #------------------------------------
                # Match DIN2 Port
                #------------------------------------
                DIN2 = re.search(DIN2_REGEX, parsed_f[i])
                if (DIN2):
                    DIN2_port = re.split(r"\)", DIN2.group())[0].replace(" ", "")

                else:
                    exit("DIN2 port doesn't exist!")
                #------------------------------------
                # Match Z/ZN Port
                #------------------------------------
                Z = re.search(Z_REGEX, parsed_f[i])
                if (Z):
                    Z_port = re.split(r"\(|,", Z.group())[1].replace(" ", "")
                    #print (Z_port)

                    gate_type_bench = ""
                    if ("nand" in gate_type):
                        gate_type_bench = "NAND"
                    elif ("xor" in gate_type):
                        gate_type_bench = "XOR"
                    elif ("nor" in gate_type):
                        gate_type_bench = "NOR"
                    elif ("or" in gate_type):
                        gate_type_bench = "OR"
                    elif ("and" in gate_type):
                        gate_type_bench = "AND"
                    of.write(Z_port+" = "+gate_type_bench+"("+DIN1_port+", "+DIN2_port+")\n")
                else:
                    exit("Z/ZN port doesn't exist!")

    of.close()
    f.close()

    fo = open(latchnamePath+"_latchname2Q_remove_LD", "w")
    for key, val in latchname2Q.items():
        fo.write(f'{key}:{val}\n')

    fo.close()

def main(argv):
    translate(argv)

def clean_verilog(benchname, filename):
    fi=open(filename, 'r')
    clean_bench=benchname+'_clean_remove_LD.v'
    fo=open(clean_bench, 'w')

    for line in fi:
        if 'module latchdr(ENA, D, R, Q);' in line or 'module fflopd' in line or 'module latchd' in line:
            break

        fo.write(line)
    fo.close()
    return clean_bench


if __name__ == "__main__":
    for idx, filepath in enumerate(glob.glob( './*')):
        if '_locked_netlist_removed.v' in filepath:
            x = re.findall(r"^.\/([A-Za-z0-9]+)_locked_netlist_removed\.v$", filepath)
            #print (x[0])
            clean_bench=clean_verilog(x[0], filepath)
            translate(x[0], clean_bench)

