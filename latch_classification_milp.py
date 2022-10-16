#!/usr/bin/env python3.7

import sys
import math
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
import os
import time

start_time = time.time()

####################### settings #########################
tb = sys.argv[1]

sol_num = 1 # number of solutions -- any integers greater than 0

class_2 = 0 # 1=based on 2-level phase 1 result  0=based on 3-level phase 1 result

set_time_limit = 1  # 1=set time limit for Gurobi   0=don't set time limit for Gurobi
time_limit = 3600 # limit for Gurobi runtime -- unit: second

ML_possibilities_reinforced = 1 # 1=yes 0=no
val_reinforced = 1 # enter any number greater than 0. larger value means trust phase 1 result more

file_in_folder = 'input'    # input file name
file_out_folder = 'output'  # output file name
##########################################################

P_LD = []
P_DD = []
P_MS = []
P_M = []
P_S = []
latch_type = []
latch_name = []

if (class_2 == 1):
    f = open('./'+file_in_folder+'/2_class/'+tb+"_softmaxprobs_remove_LD", "r")
    lines_softmaxprobs = f.readlines()
    f.close()
    for line in lines_softmaxprobs:
        if (line): # if this line is not empty (the blank last line)
            possibility = line.split()
            latch_name.append(possibility[0][:-1])
            P_MS.append(possibility[1])
            P_DD.append(possibility[2]) 
            latch_type.append(possibility[3])        
else: 
    f = open('./'+file_in_folder+'/3_class/'+tb+"_softmaxprobs_remove_LD", "r")
    lines_softmaxprobs = f.readlines()
    f.close()
    for line in lines_softmaxprobs:
        if (line): # if this line is not empty (the blank last line)
            possibility = line.split()
            latch_name.append(possibility[0][:-1])
            P_M.append(possibility[1])
            P_S.append(possibility[2])
            P_DD.append(possibility[3])
            latch_type.append(possibility[4]) 
latch_cnt = len(latch_name)
color_start = latch_cnt*3


latch2Q = {}    # key = name in result.txt      value = name in softmaxprobs
f = open('./'+file_in_folder+'/'+tb+"_latchname2Q_remove_LD", "r")
lines_latchname2Q = f.readlines()
f.close()
latch2Q_cnt = 0
for line in lines_latchname2Q:
    if (line):
        latch2Q_cnt += 1
        latch2Q[line.split(':')[1].strip()] = line.split(':')[0]

########################### process _result.txt ###########################
latch_name_result_txt = []
latch_type_result_txt = []
LD_type_result_txt = []

# these lists store the latch name in result.txt 
notLD_but_LD = []
isLD_but_not_LD = []
not_in_softmaxprobs = []

f = open('./'+file_in_folder+'/'+tb+"_result.txt", "r")
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


PI_fanout = []
PO_fanin = []
f = open('./'+file_in_folder+'/'+tb+"_PIPO_remove_LD.txt", "r")
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
            m.addConstr((T[i,0]+T[i,1])*C[i] == 0, name=constraint_3)
        
        # Set objective
        # objective is the expression that we want to get max/ min
        if (class_2 == 0):  # 3 class: Master  Slave  Delay decoy
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

  
    # add basic constraints
    m.addConstrs((T.sum(i, '*') == 1 for i in range(latch_cnt)), name='constraint_tag_sum')
    m.addConstrs((T[i,0]*(1-C[i]) == 0 for i in range(latch_cnt)), name='constraint_M')
    m.addConstrs((T[i,1]*C[i] == 0 for i in range(latch_cnt)), name='constraint_S')
      
                
    # add coloring constraints
    f = open('./'+file_in_folder+'/'+tb+"_allpaths", "r")
    lines_allpaths = f.readlines()
    f.close()
    
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
    
    for solcnt in range (m.SolCount):
        
        m.Params.SolutionNumber = solcnt    # the index of solution in the pool that we want to refer to
        # only write detailed information to output files when number of solutions is less than or equal to 1k
        if (sol_num <= 1000):
            print ("solution:", solcnt)
            out_file += "solution:" + str(solcnt) + '\n'
            type_file += "solution:" + str(solcnt) + '\n'
        if (sol_num <= 1000):
            print('Obj: %g' % m.PoolObjVal) # when we want to output the optimized object value
            out_file += 'Obj: ' + str(m.PoolObjVal) + '\n'
        sol_output = []
        
        for v in m.getVars():
            if (sol_num <= 1000):
                print('%s %g' % (v.varName, v.Xn))
                out_file += ('%s %g\n' % (v.varName, v.Xn))
            sol_output.append(str(v.varName) + ' ' + str(v.Xn))
        
        num_error = 0
        
        # verify this solution
        final_result = {}
        for j in range (latch_cnt_result_txt):
            latch2Q_key = latch_name_result_txt[j] + '_qi_reg'         
            
            if (LD_type_result_txt[j] == 'isLD'):   # classified as LD
                final_result[latch_name_result_txt[j]] = 'LATCH_LD'
                if (latch_name_result_txt[j] in isLD_but_not_LD):   
                    num_error += 1
                    if (sol_num <= 1000):    
                        print ('MISMATCH for ' + latch_name_result_txt[j] + ':\tground truth ' + latch_type_result_txt[j] + '\tGurobi LD')
                        out_file += 'MISMATCH for ' + latch_name_result_txt[j] + ':\tground truth ' + latch_type_result_txt[j] + '\tGurobi LD\n'
            else:   #if (LD_type_result_txt[j] == 'notLD'): 
                if (latch_name_result_txt[j] in not_in_softmaxprobs): # classified as DD
                    if (latch_name_result_txt[j] not in notLD_and_not_in_softmaxprobs):
                        notLD_and_not_in_softmaxprobs.append(latch_name_result_txt[j])  # consider these latches as dont care latches
                    final_result[latch_name_result_txt[j]] = 'LATCH_DD (DD after LD, dont care)'
                    if (latch_type_result_txt[j] != 'LATCH_DD'):
                        num_error += 1
                        if (latch_name_result_txt[j] not in notLD_and_not_in_softmaxprobs_and_notDD):
                            notLD_and_not_in_softmaxprobs_and_notDD.append(latch_name_result_txt[j])
                        if (sol_num <= 1000):    
                            print ('MISMATCH for ' + latch_name_result_txt[j] + ':\tground truth ' + latch_type_result_txt[j] + '\tGurobi DD')
                            out_file += 'MISMATCH for ' + latch_name_result_txt[j] + ':\tground truth ' + latch_type_result_txt[j] + '\tGurobi DD\n'
                elif (latch_name_result_txt[j] in latch2Q or latch2Q_key in latch2Q): # use Gurobi result as final result
                    if (latch_name_result_txt[j] in latch2Q):   # the latch exists in latch2Q always exist in softmaxprobs
                        i = latch_name.index(latch2Q[latch_name_result_txt[j]])
                    elif (latch2Q_key in latch2Q):
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
                        if (sol_num <= 1000):    
                            print ('MISMATCH for ' + latch_name_result_txt[j] + ':\tground truth ' + latch_type_result_txt[j] + '\tGurobi ' + Gurobi_type)
                            out_file += 'MISMATCH for ' + latch_name_result_txt[j] + ':\tground truth ' + latch_type_result_txt[j] + '\tGurobi ' + Gurobi_type + '\n'
                else:
                    print ('ERROR: ' + latch_name_result_txt[j] + ' falls into none of the cases.')
                    final_result[latch_name_result_txt[j]] = 'ERROR'
            
            if (sol_num < 1000):
                type_file += (latch_name_result_txt[j] + '\t' + final_result[latch_name_result_txt[j]] + '\n')
                        
                            
        accuracy = float((latch_cnt_real - num_error)/latch_cnt_real)
        
        if (accuracy_highest < accuracy):
            accuracy_highest = accuracy
            accuracy_highest_ind = solcnt 
        
        acc_value = str(accuracy*100) + '%'
        
        if acc_value not in accuracy_list:
            accuracy_list.append(acc_value)
        
        if (sol_num <= 1000):
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
print (isLD_but_not_LD)
print ("notLD but ground truth is LATCH_LD: " + str(len(notLD_but_LD)))
print (notLD_but_LD)
print ("notLD and does not exist in softmaxprobs and notDD: " + str(len(notLD_and_not_in_softmaxprobs_and_notDD)))
print (notLD_and_not_in_softmaxprobs_and_notDD)


if (class_2 == 1):
    class_type = 'class2_'
else:
    class_type = 'class3_'
if (sol_num <= 1000):  
    f = open('./' + file_out_folder + '/'+tb+"_gurobi_high_report_"+str(class_type)+str(sol_num), "w")
    f.writelines(out_file)
    f.close()
    f = open('./' + file_out_folder + '/'+tb+"_gurobi_high_coloring_result_"+str(class_type)+str(sol_num), "w")
    f.writelines(type_file)
    f.close()
else:
    for acc in accuracy_list:
        out_file += acc + '\n'
    out_file += 'highest accuracy: ' + str(accuracy_highest) + '\n'
    out_file += 'highest accuracy index: ' + str(accuracy_highest_ind) + '\n'
    
    f = open('./' + file_out_folder + '/'+tb+"_gurobi_high_report_"+str(class_type)+str(sol_num), "w")
    f.writelines(out_file)
    f.close()
    f = open('./' + file_out_folder + '/'+tb+"_gurobi_high_coloring_result_"+str(class_type)+str(sol_num), "w")
    f.writelines(type_file)
    f.close()

# report the runtime
temp = time.time() - start_time
hours = temp//3600
temp = temp - 3600*hours
minutes = temp//60
seconds = temp - 60*minutes
print('Total time cost = %d:%d:%d' %(hours,minutes,seconds))
