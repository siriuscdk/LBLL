import sys
import math
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
import os
import time
import numpy as np

####################### settings #########################
sol_num = 10000 # number of solutions -- any integers greater than 0

set_time_limit = 0  # 1=set time limit for Gurobi   0=don't set time limit for Gurobi
time_limit = 3600 # limit for Gurobi runtime -- unit: second

ML_possibilities_reinforced = 1 # 1=yes 0=no
val_reinforced = 1 # enter any number greater than 0. larger value means trust phase 1 result more

test_groundtruth = 0

write_file_en = 0
print_info = 0
print_key = 1

class_cnt = 2
file_in_folder = "20221014_final/"
# file_in_folder = './one_node_'+str(class_cnt)+'_level/'    # input file name
file_out_folder = '20221014_final/output_for_Gurobi'  # output file name
##########################################################


def gurobi_opt(tb, sol_idx_forkey):
    start_time = time.time()
    
    P_LD = []
    P_DD = []
    P_MS = []
    P_M = []
    P_S = []
    latch_type = []
    latch_name = []
    
    
    ########################### process _softmaxprobs ###########################
    f = open(file_in_folder+'/'+str(class_cnt)+'_all_softmaxprobs/'+tb+"_softmaxprobs_remove_LD", "r")
    lines_softmaxprobs = f.readlines()
    f.close()
    for line in lines_softmaxprobs:
        if (line): # if this line is not empty (the blank last line)
            possibility = line.split()
            latch_name.append(possibility[0][:-1])
            
            if (class_cnt == 3):
                P_M.append(possibility[1])
                P_S.append(possibility[2])
                P_DD.append(possibility[3])
                latch_type.append(possibility[4]) 
            else:   
                P_MS.append(possibility[1])
                P_DD.append(possibility[2]) 
                latch_type.append(possibility[3])        
                
    latch_cnt = len(latch_name)
    color_start = latch_cnt*3
    ################################################################################


    ########################### process _latchname2Q_remove_LD ###########################
    latch2Q = {}    # key = name in result.txt      value = name in softmaxprobs
    f = open(file_in_folder+'/post_remove_LD/'+tb+"_latchname2Q_remove_LD", "r")
    lines_latchname2Q = f.readlines()
    f.close()
    latch2Q_cnt = 0
    for line in lines_latchname2Q:
        if (line):
            latch2Q_cnt += 1
            latch2Q[line.split(':')[1].strip()] = line.split(':')[0]
    ################################################################################

    
    ########################### process _latchname2Q ##############################
    latch2Q_bef = {}    # key = name in result.txt      value = name in softmaxprobs
    f = open(file_in_folder+'/pre_remove_LD/'+tb+"_latchname2Q", "r")
    lines_latchname2Q_bef = f.readlines()
    f.close()
    latch2Q_bef_cnt = 0
    for line in lines_latchname2Q_bef:
        if (line):
            latch2Q_bef_cnt += 1
            latch2Q_bef[line.split(':')[1].rstrip().rstrip("_qi_reg")] = line.split(':')[0]
    # print (latch2Q_bef)
    ################################################################################
    
    
    ########################### process _result.txt ###########################
    latch_name_result_txt = []
    latch_type_result_txt = []
    LD_type_result_txt = []

    # these lists store the latch name in result.txt 
    notLD_but_LD = []
    isLD_but_not_LD = []
    not_in_softmaxprobs = []

    f = open(file_in_folder+'/all_results/'+tb+"_result.txt", "r")
    lines_result_txt = f.readlines()
    f.close()
    for line in lines_result_txt:
        if (line):
            line_split = line.split()
            latch_name_result_txt.append(line_split[0][:-1])
            latch_type_result_txt.append(line_split[1])
            LD_type_result_txt.append(line_split[2])
            
            if (line_split[2] == 'notLD' and line_split[1] == 'LATCH_LD'):
                notLD_but_LD.append(line_split[0][:-1])
            if (line_split[2] == 'isLD' and line_split[1] != 'LATCH_LD'):
                isLD_but_not_LD.append(line_split[0][:-1])
                
            latch2Q_key = line_split[0][:-1] + '_qi_reg'
            if (line_split[0][:-1] in latch2Q):
                if (latch2Q[line_split[0][:-1]] not in latch_name):
                    not_in_softmaxprobs.append(line_split[0][:-1])
            elif (latch2Q_key in latch2Q):
                if (latch2Q[latch2Q_key] not in latch_name):
                    not_in_softmaxprobs.append(line_split[0][:-1])
            else:
                not_in_softmaxprobs.append(line_split[0][:-1])
    latch_cnt_result_txt = len(latch_name_result_txt)
    ################################################################################
    latch_cnt_real = latch_cnt_result_txt

    
        
    ########################### process latch2PO.txt ##############################
    lat_key_idx = []
    if (print_key):
        f = open(f"{file_in_folder}/latch2PO/{tb}_latch2PO.txt", "r")
        lines_keyidx = f.readlines()
        f.close()

        for idx in range (0,len(lines_keyidx),2):
            if (lines_keyidx[idx]):
                lat_key_idx.append(lines_keyidx[idx].split(":")[0])
    ################################################################################
    
    
    
    
    ########################### process _latchname2Q_remove_LD ###########################
    PI_fanout = []
    PO_fanin = []
    f = open(file_in_folder+'/output_L2PIPO/'+tb+"_PIPO_remove_LD.txt", "r")
    lines_PIPO = f.readlines()
    f.close()
    for line in lines_PIPO:
        if (line):
            if (line.split()[1] == 'True'):
                PI_fanout.append(1) # the element in the list PI_fanout/ PO_fanin is 'int'
            else:
                PI_fanout.append(0)
            if (line.split()[2] == 'True'):
                PO_fanin.append(1)
            else:
                PO_fanin.append(0)
    ################################################################################
    

    try:

        # Create a new Gurobi model
        m = gp.Model("LC")
        if (sol_num == 1):
            m.setParam(GRB.Param.PoolSearchMode, 0)
        else:
            m.setParam(GRB.Param.PoolSearchMode, 2)
            m.setParam(GRB.Param.PoolSolutions, sol_num)
        
        if (set_time_limit == 1):
            m.setParam(GRB.Param.TimeLimit, time_limit)
        
        # Create variables
        # variable type: BINARY
        T = m.addVars(latch_cnt, 3, vtype=GRB.BINARY, name='T') # T[][0]=1 -> M  T[][1]=1 -> S  T[][2]=1 -> DD 
        C = m.addVars(latch_cnt, vtype=GRB.BINARY, name='C')   # C=1 colored as M      C=0 colored as S
        
        
        # add latch boundary constraints
        expr = gp.LinExpr()
        for i in range (latch_cnt):
            constraint_1 = 'constraint_PI_fanout_PO_fanin_' + str(i)
            constraint_2 = 'constraint_PO_fanin_DD_' + str(i)
            constraint_3 = 'constraint_PO_fanin_MS_' + str(i)
            # add constraints to the fanin of PO and fanout of PI
            if ((PI_fanout[i] == 1) and (PO_fanin[i] == 1)):
                m.addConstr(T[i,2]*(1-C[i]) == 1, name=constraint_1)
            elif (PO_fanin[i] == 1):
                m.addConstr(T[i,2]*C[i] == 0, name=constraint_2)
                m.addConstr(T[i,0] == 0, name=constraint_3)
            
            # Set objective
            # objective is the expression that we want to get max/ min
            if (class_cnt == 3):
                if (float(P_M[i]) >= 0.99):
                    PM_final = float(P_M[i])+1
                elif (float(P_M[i]) < 0.01):
                    PM_final = float(P_M[i])-1
                else:
                    PM_final = float(P_M[i])
                if (float(P_S[i]) >= 0.99):
                    PS_final = float(P_S[i])+1
                elif (float(P_S[i]) < 0.01):
                    PS_final = float(P_S[i])-1
                else:
                    PS_final = float(P_S[i])
                if (float(P_DD[i]) >= 0.99):
                    PDD_final = float(P_DD[i])+1
                elif (float(P_DD[i]) < 0.01):
                    PDD_final = float(P_DD[i])-1
                else:
                    PDD_final = float(P_DD[i])
                if (ML_possibilities_reinforced == 1):
                    expr += T[i,0]*PM_final + T[i,1]*PS_final + T[i,2]*PDD_final
                else:
                    expr += T[i,0]*float(P_M[i]) + T[i,1]*float(P_S[i]) + T[i,2]*float(P_DD[i])
            
            else:  # 2 class: Master&Slave  Delay decoy
                if (float(P_MS[i]) >= 0.99):
                    PMS_final = float(P_MS[i])+val_reinforced
                elif (float(P_MS[i]) < 0.01):
                    PMS_final = float(P_MS[i])-val_reinforced
                else:
                    PMS_final = float(P_MS[i])
                if (float(P_DD[i]) >= 0.99):
                    PDD_final = float(P_DD[i])+val_reinforced
                elif (float(P_DD[i]) < 0.01):
                    PDD_final = float(P_DD[i])-val_reinforced
                else:
                    PDD_final = float(P_DD[i])
                if (ML_possibilities_reinforced == 1):
                    expr += T[i,0]*PMS_final + T[i,1]*PMS_final + T[i,2]*PDD_final
                else:
                    expr += T[i,0]*float(P_MS[i]) + T[i,1]*float(P_MS[i]) + T[i,2]*float(P_DD[i])
       
        m.setObjective(expr, GRB.MAXIMIZE)

      
        # add fundamental constraints
        m.addConstrs((T.sum(i, '*') == 1 for i in range(latch_cnt)), name='constraint_tag_sum')
        m.addConstrs((T[i,0]*(1-C[i]) == 0 for i in range(latch_cnt)), name='constraint_M')
        m.addConstrs((T[i,1]*C[i] == 0 for i in range(latch_cnt)), name='constraint_S')
        
        
        
        f = open(file_in_folder+'/output_allpaths/'+tb+"_allpaths", "r")
        lines_allpaths = f.readlines()
        f.close()     
                    
        # add coloring constraints
        cnt = 0
        for line in lines_allpaths:
            if (line):            
                # find the latch index of i, j in the path i-> j
                latch_i_index = latch_name.index(line.split()[0])
                latch_j_index = latch_name.index(line.split()[1]) 
                
                constraint_1 = 'constraint_MM_' + str(cnt)
                constraint_2 = 'constraint_SS_' + str(cnt)
                constraint_3 = 'constraint_LD_DD_' + str(cnt)
                constraint_4 = 'constraint_DD_MS_' + str(cnt)
                
                m.addConstr((2 - T[latch_i_index,0] - T[latch_j_index,0]) >= 1, name=constraint_1)
                m.addConstr((2 - T[latch_i_index,1] - T[latch_j_index,1]) >= 1, name=constraint_2)
                m.addConstr(1 - T[latch_j_index,2] + (1- C[latch_j_index] + C[latch_i_index])*(1- C[latch_i_index] + C[latch_j_index]) >= 1, name=constraint_3)
                m.addConstr(1 - T[latch_i_index,2] + (1-T[latch_j_index,0])*(1-T[latch_j_index,1]) + C[latch_j_index]*(1-C[latch_i_index]) + C[latch_i_index]*(1-C[latch_j_index]) >= 1, name=constraint_4)
                
                cnt += 1
        
        
        # add constraints for ground truth checking
        if (test_groundtruth == 1):
            for lat_idx in range (latch_cnt):
                contraint_groundtruth_name = "ground_truth["+str(lat_idx)+"]"
                if (latch_type[lat_idx] == 'LATCH_L0'):
                    m.addConstr(T[lat_idx,0] == 1, name=contraint_groundtruth_name)
                elif (latch_type[lat_idx] == 'LATCH_L1'):
                    m.addConstr(T[lat_idx,1] == 1, name=contraint_groundtruth_name)
                elif (latch_type[lat_idx] == 'LATCH_DD'):
                    m.addConstr(T[lat_idx,2] == 1, name=contraint_groundtruth_name) 
                

        # Optimize model
        m.optimize()    
        m.write('latch_classification.lp')
        
        out_file = ''
        type_file = 'ORIGINAL SOLUTION:\n'
        notLD_and_not_in_softmaxprobs = []
        notLD_and_not_in_softmaxprobs_and_notDD = []
        accuracy_list = []
        accuracy_highest = 0
        accuracy_highest_ind = 0
        num_error_min = 99999
        
        matchkey_max = 0
        best_key_sol = 99999
        
        for solcnt in range (m.SolCount):
            
            if (print_key and solcnt == sol_idx_forkey):
                key_list = []
                for idx in range (len(lat_key_idx)*2):
                    key_list.append(0)
            
            m.Params.SolutionNumber = solcnt    # the index of solution in the pool that we want to refer to
            # only write detailed information to output files when number of solutions is less than or equal to 1k
            if (write_file_en):
                #print ("solution:", solcnt)
                out_file += "solution:" + str(solcnt) + '\n'
                type_file += "solution:" + str(solcnt) + '\n'
            if (write_file_en):
                #print('Obj: %g' % m.PoolObjVal) # when we want to output the optimized object value
                out_file += 'Obj: ' + str(m.PoolObjVal) + '\n'
            sol_output = []
            
            for v in m.getVars():
                if (write_file_en):
                    #print('%s %g' % (v.varName, v.Xn))
                    out_file += ('%s %g\n' % (v.varName, v.Xn))
                sol_output.append(str(v.varName) + ' ' + str(v.Xn))
            
            num_error = 0
            
            # verify this solution
            final_result = {}
            dont_care = []
            
            match_cnt = 0
            
            for j in range (latch_cnt_result_txt):
                latch2Q_key = latch_name_result_txt[j] + '_qi_reg'         
                
                if (LD_type_result_txt[j] == 'isLD'):   # classified as LD in phase1, this latch will not exist in softmaxprobs
                    final_result[latch_name_result_txt[j]] = 'LATCH_LD'
                    if (latch_name_result_txt[j] in isLD_but_not_LD): # classified as LD in phase1, but ground truth is not LD
                        num_error += 1
                        if (print_info):    
                            print ('MISMATCH for ' + latch_name_result_txt[j] + ':\tground truth ' + latch_type_result_txt[j] + '\tGurobi LD' + " for solution " + str(solcnt))
                            # out_file += 'MISMATCH for ' + latch_name_result_txt[j] + ':\tground truth ' + latch_type_result_txt[j] + '\tGurobi LD\n'
                else:   #if (LD_type_result_txt[j] == 'notLD'): # classified as notLD in phase1
                    if (latch_name_result_txt[j] in not_in_softmaxprobs): # does not exist in softmaxprobs
                        if (latch_name_result_txt[j] not in notLD_and_not_in_softmaxprobs): # classified as notLD and exists in softmaxprobs -> some latches after LD -> don't care
                            notLD_and_not_in_softmaxprobs.append(latch_name_result_txt[j])  # consider these latches as dont care latches
                        final_result[latch_name_result_txt[j]] = 'LATCH_DD' # (DD after LD, dont care)
                        dont_care.append(latch_name_result_txt[j])
                        if (latch_type_result_txt[j] != 'LATCH_DD'): # ground truth is not DD but does not exist in softmaxprobs, as we assume them to be DD after LD and was removed together with LD
                            num_error += 1
                            if (latch_name_result_txt[j] not in notLD_and_not_in_softmaxprobs_and_notDD):
                                notLD_and_not_in_softmaxprobs_and_notDD.append(latch_name_result_txt[j])
                            if (print_info):    
                                print ('MISMATCH for ' + latch_name_result_txt[j] + ':\tground truth ' + latch_type_result_txt[j] + '\tGurobi DD' + " for solution " + str(solcnt))
                                # out_file += 'MISMATCH for ' + latch_name_result_txt[j] + ':\tground truth ' + latch_type_result_txt[j] + '\tGurobi DD\n'
                    elif (latch_name_result_txt[j] in latch2Q or latch2Q_key in latch2Q): # exists in softmaxprobs, use Gurobi result as final result
                        if (latch_name_result_txt[j] in latch2Q): # the latch exists in latch2Q always exist in softmaxprobs
                            i = latch_name.index(latch2Q[latch_name_result_txt[j]])
                        elif (latch2Q_key in latch2Q): # if cannot find the name in results.txt in latch2Q, try name concatenate with "_qi_reg"
                            i = latch_name.index(latch2Q[latch2Q_key])
                        TAG_M = round(float(sol_output[i*3].split()[1]))
                        TAG_S = round(float(sol_output[i*3+1].split()[1]))
                        TAG_DD = round(float(sol_output[i*3+2].split()[1]))                 
                        COLOR = round(float(sol_output[color_start+i].split()[1]))
                        if (TAG_M == 1):
                            Gurobi_type = 'LATCH_L0'
                        elif (TAG_S == 1):
                            Gurobi_type = 'LATCH_L1'       
                        else:
                            Gurobi_type = 'LATCH_DD'
                        final_result[latch_name_result_txt[j]] = Gurobi_type
                        if (latch_type_result_txt[j] != Gurobi_type):
                            num_error += 1
                            if (print_info):    
                                print ('MISMATCH for ' + latch_name_result_txt[j] + ':\tground truth ' + latch_type_result_txt[j] + '\tGurobi ' + Gurobi_type + " for solution " + str(solcnt))
                                # out_file += 'MISMATCH for ' + latch_name_result_txt[j] + ':\tground truth ' + latch_type_result_txt[j] + '\tGurobi ' + Gurobi_type + '\n'
                    else:
                        print ('ERROR: ' + latch_name_result_txt[j] + ' falls into none of the cases.')
                        final_result[latch_name_result_txt[j]] = 'ERROR'
                

                if (write_file_en):
                    type_file += (latch_name_result_txt[j] + '\t' + latch_type_result_txt[j] + '\t' + final_result[latch_name_result_txt[j]] + '\n')
                
                # generate key results
                # if (print_key and solcnt == sol_idx_forkey):
                if (print_key):
                    
                    #print (f"{latch_name_result_txt[j]}  {j}")
                    #print (lat_key_idx)
                    if (latch_name_result_txt[j].rstrip("_qi_reg") in lat_key_idx):
                        
                        if (solcnt == sol_idx_forkey):
                            idx = lat_key_idx.index(latch_name_result_txt[j].rstrip("_qi_reg"))
                         
                        if (final_result[latch_name_result_txt[j]] == 'LATCH_L0'):
                            if (solcnt == sol_idx_forkey):
                                key_list[2*idx] = 1
                                key_list[2*idx+1] = 0
                            if (latch_type_result_txt[j] == "LATCH_L0"):
                                match_cnt += 2
                            elif (latch_type_result_txt[j] == "LATCH_L1" or latch_type_result_txt[j] == "LATCH_LD"):
                                match_cnt += 1
                            
                        elif (final_result[latch_name_result_txt[j]] == 'LATCH_L1'):
                            if (solcnt == sol_idx_forkey):
                                key_list[2*idx] = 1
                                key_list[2*idx+1] = 1
                            if (latch_type_result_txt[j] == "LATCH_L1"):
                                match_cnt += 2
                            elif (latch_type_result_txt[j] == "LATCH_L0" or latch_type_result_txt[j] == "LATCH_DD"):
                                match_cnt += 1
                            
                        elif (final_result[latch_name_result_txt[j]] == 'LATCH_DD'):
                            if (solcnt == sol_idx_forkey):
                                key_list[2*idx] = 0
                                key_list[2*idx+1] = 1
                            if (latch_type_result_txt[j] == "LATCH_DD"):
                                match_cnt += 2
                            elif (latch_type_result_txt[j] == "LATCH_L1" or latch_type_result_txt[j] == "LATCH_LD"):
                                match_cnt += 1
                            
                        elif (final_result[latch_name_result_txt[j]] == 'LATCH_LD'):
                            if (solcnt == sol_idx_forkey):
                                key_list[2*idx] = 0
                                key_list[2*idx+1] = 0
                            if (latch_type_result_txt[j] == "LATCH_LD"):
                                match_cnt += 2
                            elif (latch_type_result_txt[j] == "LATCH_L0" or latch_type_result_txt[j] == "LATCH_DD"):
                                match_cnt += 1
            # print (f"match cnount = {match_cnt} for sol {solcnt}")
            
            if (match_cnt > matchkey_max):
                matchkey_max = match_cnt
                best_key_sol = solcnt
            
            accuracy = float((latch_cnt_real - num_error)/latch_cnt_real)
            
            if (accuracy_highest < accuracy):
                accuracy_highest = accuracy
                accuracy_highest_ind = solcnt 
                
            if (num_error_min > num_error):
                num_error_min = num_error
            
            acc_value = str(accuracy_highest*100) + '%'
            
            if acc_value not in accuracy_list:
                accuracy_list.append(acc_value)
            
            if (write_file_en):
                print ('accuracy = ' + acc_value)
                out_file += 'accuracy = ' + acc_value + '\n'
            

    # error handle in gurobi
    # report the error
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))

    except AttributeError:
        print('Encountered an attribute error')


    # report the size of the circuit
    print ('circuit latch count = ' + str(latch_cnt_real))
    print (accuracy_list)

    # report the classification latch in phase 1
    print ("isLD but ground truth is not LATCH_LD: " + str(len(isLD_but_not_LD)))
    #print (isLD_but_not_LD)
    print ("notLD but ground truth is LATCH_LD: " + str(len(notLD_but_LD)))
    #print (notLD_but_LD)
    print ("notLD and does not exist in softmaxprobs and notDD: " + str(len(notLD_and_not_in_softmaxprobs_and_notDD)))
    #print (notLD_and_not_in_softmaxprobs_and_notDD)
    print (f"Dont care latches: {len(dont_care)}")
    # print (f"Dont care latches {dont_care}")

    hard_error_list = []
    for lat in isLD_but_not_LD:
        if lat not in hard_error_list:
            hard_error_list.append(lat)
    for lat in notLD_but_LD:
        if lat not in hard_error_list:
            hard_error_list.append(lat)
    for lat in notLD_and_not_in_softmaxprobs_and_notDD:
        if lat not in hard_error_list:
            hard_error_list.append(lat)
    print (f"The # of latch errors that cannot be fixed by ILP = {len(hard_error_list)}")
    print (f"The # of total latches = {latch_cnt_real}")
    print (f"The # of min error = {num_error_min}")
    print (f"The value of highest accuracy = {accuracy_highest*100}%")
    print (f"The index of highest solution = {accuracy_highest_ind}")
    
    
    # print predicted key
    if (print_key):
        # clear file
        f = open(f"{file_out_folder}/{tb}_predicted_key_class{class_cnt}_sol{sol_num}", "w")
        f.close()
        f = open(f"{file_out_folder}/{tb}_predicted_key_class{class_cnt}_sol{sol_num}", "a")
        for every_key in key_list:
            f.write(f"{every_key} ")
        f.close()
    
    # compare with correct key
    if (print_key):
        '''
        f = open(f"{file_in_folder}/correct_key/{tb}_correct_key.txt", "r")
        line_key = f.readlines()[0]
        correct_key_list = line_key.split()
        f.close()
        for idx in range(len(correct_key_list)):
            correct_key_list[idx] = int(correct_key_list[idx])
        
        if (len(correct_key_list) != len(key_list)):
            print (f"Error: key list length mismatch")
        #print (f"correct key length = {len(correct_key_list)}  predicted key length = {len(key_list)}   latch cnt = {latch_cnt_real}")
        #print (correct_key_list)
        #print (key_list)
        
        correct_key_list = np.array(correct_key_list)
        key_list = np.array(key_list)
        '''
        # match_cnt = np.sum(correct_key_list == key_list)
        print (f"key match # = {matchkey_max}  key accuracy = {(matchkey_max/(len(lat_key_idx)*2))*100}% for sol {best_key_sol}")
    
   
    
    if (write_file_en):
        for acc in accuracy_list:
            out_file += acc + '\n'
        out_file += 'highest accuracy: ' + str(accuracy_highest) + '\n'
        out_file += 'highest accuracy index: ' + str(accuracy_highest_ind) + '\n'
            
        f = open('./' + file_out_folder + '/'+tb+"_gurobi_high_report_"+str(sol_num), "w")
        f.writelines(out_file)
        f.close()
        f = open('./' + file_out_folder + '/'+tb+"_gurobi_high_coloring_result_"+str(sol_num), "w")
        f.writelines(type_file)
        f.close()

    # report the runtime
    temp = time.time() - start_time
    hours = temp//3600
    temp = temp - 3600*hours
    minutes = temp//60
    seconds = temp - 60*minutes
    print('Total time cost = %d:%d:%d' %(hours,minutes,seconds))
    
    if (print_key):
        return matchkey_max, (len(lat_key_idx)*2), num_error_min, latch_cnt_real, len(isLD_but_not_LD), len(notLD_but_LD), len(notLD_and_not_in_softmaxprobs_and_notDD), len(hard_error_list), accuracy_highest_ind, best_key_sol
    else:
        return 0, 0, num_error_min, latch_cnt_real, len(isLD_but_not_LD), len(notLD_but_LD), len(notLD_and_not_in_softmaxprobs_and_notDD), len(hard_error_list), accuracy_highest_ind, 0
        
    
def main(): 
    #circuit_list = ['s298','s9234','s13207','s15850','s35932','s38417','s38584','b03','b04','b07','b11','b12','b13','b14','b15','b17','b18','b19','b20','b21','b22']
    circuit_list = ['s298','s9234','s13207','s15850','s35932','s38417','s38584','b03','b04','b07','b11','b12','b13','b14','b15','b17','b20','b21','b22']
    
   
    sum_list = []
    for i in range (10):
        sum_list.append([])
        
    for idx in range (len(circuit_list)):
        (correct_key_cnt, key_cnt, wrong_lat_cnt, lat_cnt, isLD_but_not_LD_cnt, notLD_but_LD_cnt, notLD_and_not_in_softmaxprobs_and_notDD_cnt, hard_error_cnt, best_idx, best_key_sol) = gurobi_opt(circuit_list[idx], 0)
        #os.system("pause")
        print (f"Info: Use #{best_key_sol} as the index for key file output")
        
        if (best_key_sol != 0):
            (correct_key_cnt, key_cnt, wrong_lat_cnt, lat_cnt, isLD_but_not_LD_cnt, notLD_but_LD_cnt, notLD_and_not_in_softmaxprobs_and_notDD_cnt, hard_error_cnt, best_idx, best_key_sol_repeat) = gurobi_opt(circuit_list[idx], best_key_sol)
            
            if (best_key_sol_repeat != best_key_sol):
                print (f"Error: The solution #id differs in 2 rounds of ILP.")
        
        sum_list[0].append(correct_key_cnt)
        sum_list[1].append(key_cnt)
        sum_list[2].append(wrong_lat_cnt)
        sum_list[3].append(lat_cnt)
        sum_list[4].append(isLD_but_not_LD_cnt)
        sum_list[5].append(notLD_but_LD_cnt)
        sum_list[6].append(notLD_and_not_in_softmaxprobs_and_notDD_cnt)
        sum_list[7].append(hard_error_cnt)
        sum_list[8].append(best_idx)
        sum_list[9].append(best_key_sol)
        
    for i in range (10):
        print (sum_list[i])
    
if __name__=='__main__':
    main()