# Executed in Python 3.6
import re, random
from collections import OrderedDict
from Ntk_Struct_PO_cmu import *


###########################################################################
# Function name: ntk_parser
# Note: Parse the .bench format netlist into Python
###########################################################################
def ntk_parser(ipt_file):
    Circuit_graph = Ntk()
    # Read a .BENCH netlist file
    num_nor=0
    ipt = open(ipt_file)
    Circuit_graph.circuit_name = ipt_file  # Give name to a netlist graph
    for line in ipt:  # Construct the graph and vertices
        line = line.strip()
        if line != "":
            if re.match(r'^#', line):
                pass
            else:
                if '#' in line:
                    line = line[:line.index('#')]
                line_syntax = re.match(r'^([A-Za-z_\$\[\]]+) ?\((.+)\)', line) # macth input/output ports
                if line_syntax:
                    if re.match(r'INPUT', line_syntax.group(1), re.IGNORECASE):
                        ipt_node = line_syntax.group(2)
                        if not ipt_node[0].isalpha():  # If the node name does not start with a letter, add a letter 'G' before it.
                            ipt_node = 'G' + ipt_node
                        if ipt_node not in Circuit_graph.object_name_list:
                            new_node = NtkObject(ipt_node)
                            Circuit_graph.add_object(new_node, 'IPT')
                            # if 'keyinput' not in ipt_node:
                            if 'key' not in ipt_node:
                                Circuit_graph.PI.append(new_node)
                            else:
                                Circuit_graph.KI.append(new_node)
                                Circuit_graph.available_key_index += 1

                    elif re.match(r'OUTPUT', line_syntax.group(1), re.IGNORECASE):
                        opt_node = line_syntax.group(2)
                        if not opt_node[0].isalpha():  # If the node name does not start with a letter, add a letter 'G' before it.
                            opt_node = 'G' + opt_node
                        if opt_node not in Circuit_graph.object_name_list:
                            #print ("add opt:", opt_node)
                            opt_node = opt_node+'_PO'
                            new_node = NtkObject(opt_node)
                            #print ("add opt name:", opt_node)
                            Circuit_graph.add_object(new_node)
                            Circuit_graph.PO.append(new_node)
                            Circuit_graph.POname.append(opt_node)

                else: # match gates
                    line_syntax = re.match(r' *([a-zA-Z0-9_\$\[\]]+) *= *([a-zA-Z0-9_\[\]]+) *\( *(.+) *\)', line)
                    left_node = line_syntax.group(1)
                    right_nodes = re.split(r' *, *', line_syntax.group(3))
                    #print (left_node, right_nodes)
                    que = line.split(' ')
                    if not left_node[0].isalpha():  # If the node name does not start with a letter, add a letter 'G' before it.
                        left_node = 'G' + left_node

                    if left_node+'_PO' in Circuit_graph.POname:
                        left_node=left_node+'_PO'

                    if left_node not in Circuit_graph.object_name_list:
                        new_node = NtkObject(left_node)
                        Circuit_graph.add_object(new_node)

                    if re.match(r'NOT', line_syntax.group(2), re.IGNORECASE):
                        Circuit_graph.find_node_by_name(left_node).gate_type = Circuit_graph.gateType['NOT']
                    elif re.match(r'NAND', line_syntax.group(2), re.IGNORECASE):
                        Circuit_graph.find_node_by_name(left_node).gate_type = Circuit_graph.gateType['NAND']
                    elif re.match(r'AND', line_syntax.group(2), re.IGNORECASE):
                        Circuit_graph.find_node_by_name(left_node).gate_type = Circuit_graph.gateType['AND']
                    elif re.match(r'XNOR', line_syntax.group(2), re.IGNORECASE):
                        Circuit_graph.find_node_by_name(left_node).gate_type = Circuit_graph.gateType['XNOR']
                    elif re.match(r'NOR', line_syntax.group(2), re.IGNORECASE):
                        Circuit_graph.find_node_by_name(left_node).gate_type = Circuit_graph.gateType['NOR']
                    elif re.match(r'XOR', line_syntax.group(2), re.IGNORECASE):
                        Circuit_graph.find_node_by_name(left_node).gate_type = Circuit_graph.gateType['XOR']
                    elif re.match(r'OR', line_syntax.group(2), re.IGNORECASE):
                        Circuit_graph.find_node_by_name(left_node).gate_type = Circuit_graph.gateType['OR']
                    elif re.match(r'DFF', line_syntax.group(2), re.IGNORECASE):
                        Circuit_graph.find_node_by_name(left_node).gate_type = Circuit_graph.gateType['DFF']
                    elif re.match(r'BUFF?', line_syntax.group(2), re.IGNORECASE):
                        Circuit_graph.find_node_by_name(left_node).gate_type = Circuit_graph.gateType['BUFF']
                    elif re.match(r'MUX', line_syntax.group(2), re.IGNORECASE):
                        Circuit_graph.find_node_by_name(left_node).gate_type = Circuit_graph.gateType['MUX']
                    elif re.match(r'LATCH_L0', line_syntax.group(2), re.IGNORECASE):
                        Circuit_graph.find_node_by_name(left_node).gate_type = Circuit_graph.gateType['LATCH_L0']
                    elif re.match(r'LATCH_L1', line_syntax.group(2), re.IGNORECASE):
                        Circuit_graph.find_node_by_name(left_node).gate_type = Circuit_graph.gateType['LATCH_L1']
                    elif re.match(r'LATCH_LD', line_syntax.group(2), re.IGNORECASE):
                        Circuit_graph.find_node_by_name(left_node).gate_type = Circuit_graph.gateType['LATCH_LD']
                    elif re.match(r'LATCH_DD', line_syntax.group(2), re.IGNORECASE):
                        Circuit_graph.find_node_by_name(left_node).gate_type = Circuit_graph.gateType['LATCH_DD']
                    # TODO: Sequential features missing
                    # if que[0] not in reg_out:
                    # 	reg_out.append(que[0])
                    else:
                        print("New Logic Element in the following line!!!")
                        print(line)
                        exit(1)
                    for node in right_nodes:
                        if not node[0].isalpha():  # If the node name does not start with a letter, add a letter 'G' before it.
                            node = 'G' + node

                        if node + '_PO' in Circuit_graph.POname:
                            node = node + '_PO'

                        if node not in Circuit_graph.object_name_list:
                            new_node = NtkObject(node)
                            Circuit_graph.add_object(new_node)
                        Circuit_graph.connect_objectives_by_name(node, left_node)
    #print (num_nor)
    ipt.close()
    return Circuit_graph


###########################################################################
# Function name: ntk_levelization
# Note: Levelize the netlist. This function must be after parsing a netlist
# or after any modification to the Ntk class object
###########################################################################
def ntk_levelization(Circuit_graph, fic_enable=False):
    Circuit_graph.simulation_starting_obj = None
    Circuit_graph.simulation_ending_obj = None
    for node in Circuit_graph.object_list:
        # if node.topo_sort_index is None:
        node.topo_sort_index = len(node.fan_in_node)

    queue = [node for node in Circuit_graph.PI + Circuit_graph.KI]

    while len(queue):
        current_node = queue[0]
        for fan_out_node in current_node.fan_out_node:
            fan_out_node.topo_sort_index -= 1
            if fan_out_node.topo_sort_index == 0:
                queue.append(fan_out_node)
        if Circuit_graph.simulation_starting_obj is None:  # Set the starting node for simulation
            Circuit_graph.simulation_starting_obj = current_node
        if len(queue) == 1:  # Set the ending node for simulation
            assert (Circuit_graph.simulation_ending_obj is None)
            Circuit_graph.simulation_ending_obj = current_node
        else:  # Point the next node to the current one
            current_node.next_node = queue[1]
        queue = queue[1:]
    for node in Circuit_graph.object_list:
        # assert(node.topo_sort_index == 0)
        if node.topo_sort_index != 0:
            print("Levelization error: %s" % node.name)
    if fic_enable:
        find_fan_in_cone(Circuit_graph)


###########################################################################
# Function name: ntk_to_bench
# Note: Output the Ntk class object to a .bench format netlist
###########################################################################
def ntk_to_bench(Circuit_graph, opt_file_path):  # Write circuit_graph to .bench file
    opt_file = open(opt_file_path, "w")
    zipped = zip([node.name for node in Circuit_graph.PI], Circuit_graph.PI)
    zipped = sorted(zipped)
    if len(zipped):  # This a circuit has any PI
        sorted_PI = list(zip(*zipped))[1]
        for node in sorted_PI:
            opt_file.write("INPUT(%s)\n" % node.name)
        for node in Circuit_graph.KI:
            opt_file.write("INPUT(%s)\n" % node.name)
        opt_file.write("\n")
    zipped = zip([node.name for node in Circuit_graph.PO], Circuit_graph.PO)
    zipped = sorted(zipped)
    if len(zipped):  # This a circuit has any PO
        sorted_PO = list(zip(*zipped))[1]
        for node in sorted_PO:
            opt_file.write("OUTPUT(%s)\n" % node.name)
        opt_file.write("\n")
    for node in Circuit_graph.object_list:
        if node.gate_type != Circuit_graph.gateType['IPT']:
            opt_file.write("%s = %s(" % (node.name, Circuit_graph.gateType_reverse[node.gate_type])),
            ipt_num = len(node.fan_in_node)
            for ipt_node in node.fan_in_node:
                ipt_num -= 1
                if ipt_num == 0:
                    opt_file.write("%s" % ipt_node.name),
                else:
                    opt_file.write("%s, " % ipt_node.name),
            opt_file.write(") \n")

    opt_file.close()


###########################################################################
# Function name: find_fan_in_cone
# Note: For each node, find all other nodes that are in the fan-in cone of
#       this node.
###########################################################################
def find_fan_in_cone(circuit_graph):  # Given a netlist, obtain the fan-in cone of each node in the netlist
    current_node = circuit_graph.simulation_starting_obj
    while current_node is not None:
        for ipt_node in current_node.fan_in_node:
            for temp in ipt_node.fan_in_cone:
                if temp not in current_node.fan_in_cone:
                    current_node.fan_in_cone.append(temp)
            if ipt_node not in current_node.fan_in_cone:
                current_node.fan_in_cone.append(ipt_node)
            if ipt_node.influence_by_key is True:
                current_node.influence_by_key = True
        if current_node.influence_by_key is None:
            if current_node in circuit_graph.KI:
                current_node.influence_by_key = True
            else:
                current_node.influence_by_key = False
        current_node = current_node.next_node


###########################################################################
# Function name: find_largest_fan_in_cone
# Note: Return the PO node with the largest fan-in cone
###########################################################################
def find_largest_fan_in_cone(circuit_graph):
    largest_node = circuit_graph.PO[0]
    for node in circuit_graph.PO:
        if len(node.fan_in_cone) > len(largest_node.fan_in_cone):
            largest_node = node
    return largest_node


###########################################################################
# Function name: reorder_netlist
# Note: Functions to reorganize the netlist in the alphabetical order.
###########################################################################
def reorder_netlist(input_file, output_file):
    circuit_graph = ntk_parser(input_file)
    ntk_to_bench(circuit_graph, output_file)


###########################################################################
# Function name: shuffle_netlist
# Note: Functions to shuffle the netlist n times.
###########################################################################
def shuffle_netlist(encrypted_netlist, output_name, n):
    for count in range(n):
        # Read a .BENCH netlist file
        ipt = open(encrypted_netlist, 'r')
        opt = open(output_name + '_' + str(count) + '.bench', 'w')
        to_be_shuffled = []
        for line in ipt:  # Construct the graph and vertices
            line = line.strip()
            if line != "":
                if re.match(r'^#', line):
                    opt.write(line + '\n')
                else:
                    if '#' in line:
                        line = line[:line.index('#')]
                    line_syntax = re.match(r'^([A-Za-z_\$]+) ?\((.+)\)', line)
                    if line_syntax:
                        if re.match(r'INPUT', line_syntax.group(1), re.IGNORECASE):
                            opt.write(line + '\n')

                        elif re.match(r'OUTPUT', line_syntax.group(1), re.IGNORECASE):
                            opt.write(line + '\n')
                    else:
                        to_be_shuffled.append(line)
        random.shuffle(to_be_shuffled)
        for new_line in to_be_shuffled:
            opt.write(new_line + '\n')
        ipt.close()
        opt.close()


###########################################################################
# Function name: seq_to_comb
# Note: Remove the DFFs in the netlist and output as a new file.
###########################################################################
def seq_to_comb(input_path, output_path):
    circuit_graph = ntk_parser(input_path)
    opt_file = open(output_path, "w")
    zipped = zip([node.name for node in circuit_graph.PI], circuit_graph.PI)
    zipped = sorted(zipped)
    if len(zipped):  # This a circuit has any PI
        sorted_PI = list(zip(*zipped))[1]
        for node in sorted_PI:
            opt_file.write("INPUT(%s)\n" % node.name)
        for node in circuit_graph.KI:
            opt_file.write("INPUT(%s)\n" % node.name)
    # If the IO of the DFF is directly the PI/PO, add buffers
    pi_node_of_concern = []
    po_node_of_concern = []
    for node in circuit_graph.object_list:
        if node.gate_type == circuit_graph.gateType['DFF']:
            if node in circuit_graph.PO:
                po_node_of_concern.append(node)
            if node.fan_in_node[0] in circuit_graph.PI:
                pi_node_of_concern.append(node.fan_in_node[0])
    for node in po_node_of_concern:
        # print('From PO: %s' % node.name)
        temp = NtkObject(node.name)
        circuit_graph.add_object(temp, 'BUFF')
        circuit_graph.remove_node_from_PO(node)
        circuit_graph.add_PO(temp)
        circuit_graph.object_name_list[circuit_graph.object_list.index(temp)] = node.name
        circuit_graph.object_name_list[circuit_graph.object_list.index(node)] = node.name + 't'
        node.name = node.name + 't'
        temp_list = []
        for opt_node in node.fan_out_node:
            temp_list.append(opt_node)
        for opt_node in temp_list:
            circuit_graph.disconnect_objectives(node, opt_node)
            circuit_graph.connect_objectives(temp, opt_node)
        circuit_graph.connect_objectives(node, temp)

    for node in pi_node_of_concern:
        print('From PI: %s' % node.name)
        temp = NtkObject(node.name)
        circuit_graph.add_object(temp, 'IPT')
        node.gate_type = circuit_graph.gateType['BUFF']
        circuit_graph.remove_node_from_PI(node)
        circuit_graph.add_PI(temp)
        circuit_graph.object_name_list[circuit_graph.object_list.index(temp)] = node.name
        circuit_graph.object_name_list[circuit_graph.object_list.index(node)] = node.name + 't'
        node.name = node.name + 't'
        temp_list = []
        for ipt_node in node.fan_in_node:
            temp_list.append(ipt_node)
        for ipt_node in temp_list:
            circuit_graph.disconnect_objectives(ipt_node, node)
            circuit_graph.connect_objectives(ipt_node, temp)
        circuit_graph.connect_objectives(temp, node)
    dff_ipt = []
    name_changed = []
    for node in circuit_graph.object_list:
        if node.gate_type == circuit_graph.gateType['DFF']:
            opt_file.write("INPUT(%s)\n" % ('DF_' + node.name))
            name_changed.append(node)
            dff_ipt.append(node.fan_in_node[0].name)
            name_changed.append(node.fan_in_node[0])
    opt_file.write("\n")

    zipped = zip([node.name for node in circuit_graph.PO], circuit_graph.PO)
    zipped = sorted(zipped)
    if len(zipped):  # This a circuit has any PO
        sorted_PO = list(zip(*zipped))[1]
        for node in sorted_PO:
            opt_file.write("OUTPUT(%s)\n" % node.name)
    for node in dff_ipt:
        opt_file.write("OUTPUT(%s)\n" % ('DF_' + node))
    opt_file.write("\n")

    for node in circuit_graph.object_list:
        if node.gate_type != circuit_graph.gateType['IPT'] and node.gate_type != circuit_graph.gateType['DFF']:
            if node in name_changed:
                opt_file.write("%s = %s(" % ('DF_' + node.name, circuit_graph.gateType_reverse[node.gate_type])),
            else:
                opt_file.write("%s = %s(" % (node.name, circuit_graph.gateType_reverse[node.gate_type])),
            ipt_num = len(node.fan_in_node)
            for ipt_node in node.fan_in_node:
                ipt_num -= 1
                if ipt_num == 0:
                    if ipt_node in name_changed:
                        opt_file.write("%s" % ('DF_' + ipt_node.name)),
                    else:
                        opt_file.write("%s" % ipt_node.name),
                else:
                    if ipt_node in name_changed:
                        opt_file.write("%s, " % ('DF_' + ipt_node.name)),
                    else:
                        opt_file.write("%s, " % ipt_node.name),
            opt_file.write(") \n")

    opt_file.close()


###########################################################################
# Function name: name_aliasing
# Note: For a given circuit_graph, we will change all intermediate nodes
#       name to [stamp + number].
###########################################################################
def name_aliasing(circuit_graph, stamp):
    for obj_index in range(len(circuit_graph.object_list)):
        if (circuit_graph.object_list[obj_index] not in circuit_graph.PI and circuit_graph.object_list[obj_index] not in circuit_graph.PO and circuit_graph.object_list[obj_index] not in circuit_graph.KI):
            new_name = stamp + str(obj_index)
            circuit_graph.object_name_list[obj_index] = stamp + str(obj_index)
            circuit_graph.object_list[obj_index].name = new_name
    return circuit_graph


###########################################################################
# Function name: ntk_extract
# Note: Extract part of the circuit graph to a new graph.
###########################################################################
def ntk_extract(original_graph, extracted_component_list, level_flag=False):
    extracted_component_names = []
    extracted_component_objects = []
    # Check if the given component list is node names (str) or node objects
    if len(extracted_component_list) > 0:
        if type(extracted_component_list[0]) == str:
            extracted_component_names = extracted_component_list
            for name in extracted_component_names:
                extracted_component_objects.append(original_graph.find_node_by_name(name))
        elif type(extracted_component_list[0]) == NtkObject:
            extracted_component_objects = extracted_component_list
            for obj in extracted_component_objects:
                extracted_component_names.append(original_graph.find_name_by_node(obj))
        else:
            print('Wrong format: \"extracted_component_list\"')
            exit(0)
    else:
        extracted_graph = Ntk()
        return extracted_graph
    # Create an empty circuit graph
    extracted_graph = Ntk()
    # Copy the extracted node to the extracted graph
    for node in extracted_component_objects:
        copy = NtkObject(node.name)
        extracted_graph.add_object(copy, extracted_graph.gateType_reverse[node.gate_type])
    # Connect the component
    for node in extracted_component_objects:
        copy = extracted_graph.find_node_by_name(node.name)
        for ipt_node in node.fan_in_node:
            if ipt_node in extracted_component_objects:  # If its fan-in node is also in the extracted component list
                # Add the connection is it is not connected yet
                if extracted_graph.find_node_by_name(ipt_node.name) not in extracted_graph.find_node_by_name(node.name).fan_in_node:
                    extracted_graph.connect_objectives_by_name(ipt_node.name, node.name)
            else:  # If the fan-in node is not in the extracted component list, make it a PI for the extracted graph
                if ipt_node.name not in extracted_graph.object_name_list:
                    temp = NtkObject(ipt_node.name)
                    Ntk.add_object(extracted_graph, temp, 'IPT')
                    extracted_graph.add_PI(temp)
                extracted_graph.connect_objectives_by_name(ipt_node.name, node.name)
        if len(node.fan_out_node) == 0:
            extracted_graph.add_PO(copy)
        else:
            for opt_node in node.fan_out_node:
                if opt_node in extracted_component_objects:  # If its fan-out node is also in the extracted component list
                    # Add the connection is it is not connected yet
                    if extracted_graph.find_node_by_name(opt_node.name) not in extracted_graph.find_node_by_name(node.name).fan_out_node:
                        extracted_graph.connect_objectives_by_name(node.name, opt_node.name)
                else:  # If the fan-out node is not in the extracted component list, make it a PO for the extracted graph
                    extracted_graph.add_PO(copy)
    if level_flag:
        ntk_levelization(extracted_graph)
    return extracted_graph


###########################################################################
# Function name: ntk_stitch
# Note: Replace part of a circuit with the given new part.
#       In this function, we assume that the order of PI and PO corresponds
#       to the ones of the extracted graph.
###########################################################################
def ntk_stitch(original_graph, extracted_graph, new_part_graph):
    # Cut the connection between the extracted part and the original graph
    for ipt_node in extracted_graph.PI:
        for opt_node in ipt_node.fan_out_node:
            # If the fan-out node is in the extracted graph
            if opt_node in extracted_graph.object_list:
                original_graph.disconnect_objectives_by_name(ipt_node.name, opt_node.name)
    po_fan_out_list = []
    replaced_po = []  # This list is to record the PO node if it will be replaced by the new part
    for opt_node in extracted_graph.PO:
        temp_list = []
        for opt_opt_node in original_graph.find_node_by_name(opt_node.name).fan_out_node:
            temp_list.append(opt_opt_node)
        for node in temp_list:
            original_graph.disconnect_objectives_by_name(opt_node.name, node.name)
        if len(temp_list) == 0:
            replaced_po.append(original_graph.find_node_by_name(opt_node.name))
        po_fan_out_list.append(temp_list)
    # Remove the internal nodes in the original graph
    for node_name in extracted_graph.object_name_list:
        if extracted_graph.find_node_by_name(node_name) not in extracted_graph.PI:
            if extracted_graph.find_node_by_name(node_name) not in extracted_graph.PO:
                original_graph.remove_object(original_graph.find_node_by_name(node_name))
    # Include the nodes of the new part to the original graph
    for node in new_part_graph.object_list:
        if node not in new_part_graph.PI:
            if node not in new_part_graph.KI:
                if node.name in original_graph.object_name_list:
                    new_part_graph.change_node_name('n' + str(original_graph.available_node_index))
                original_graph.add_object(node)
            else:
                if node.name in original_graph.object_name_list:
                    new_part_graph.change_node_name(node, 'keyinput' + str(original_graph.available_key_index))
                    original_graph.available_key_index += 1
                original_graph.add_object(node)
                original_graph.add_KI(node)
    # Connect the nodes of the new part to the original graph
    i = 0
    for node in new_part_graph.PI:
        to_be_deleted = []
        for opt_node in node.fan_out_node:
            original_graph.connect_objectives_by_name(extracted_graph.PI[i].name, opt_node.name)
            to_be_deleted.append(opt_node)
        for temp in to_be_deleted:
            new_part_graph.disconnect_objectives(node, temp)
        i += 1
    # assert len(po_fan_out_list) == len(new_part_graph.PO)
    i = 0
    j = 0
    non_deletable_po = []  # This list records all the PO nodes if they are at the boundary of the extracted part
    for node_list in po_fan_out_list:
        if len(node_list) == 0:
            po_name = replaced_po[j].name
            non_deletable_po.append(po_name)
            original_graph.remove_object(replaced_po[j])
            original_graph.change_node_name(new_part_graph.PO[i], po_name)
            original_graph.add_PO(new_part_graph.PO[i])
            j += 1
        else:
            for node in node_list:
                if node in original_graph.object_list:
                    original_graph.connect_objectives_by_name(new_part_graph.PO[i].name, node.name)
        i += 1
    # Remove the PO node of the extracted from the original
    for node in extracted_graph.PO:
        if node.name not in non_deletable_po:
            original_graph.remove_object(original_graph.find_node_by_name(node.name))
