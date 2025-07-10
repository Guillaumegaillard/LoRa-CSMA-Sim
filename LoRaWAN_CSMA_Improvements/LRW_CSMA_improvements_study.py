# -*- coding: utf-8 -*-
######################### Setup for studying LoRaWAN CSMA improvements with LoRa-CSMA-Sim ##################################
######################### https://github.com/Guillaumegaillard/LoRa-CSMA-Sim ###############################################
######################### 2025-07-09 #######################################################################################
######################### License MIT: #####################################################################################
######################### https://github.com/Guillaumegaillard/LoRa-CSMA-Sim/blob/main/LICENSE #############################
######################### Inspired from LoRaSim https://mcbor.github.io/lorasim/ ###########################################

import lora_csma_sim as simu
import topo_builder

import pickle
import json

import sys
import os

if not os.path.exists('results'):
    os.makedirs('results')


actions = {
    "build_topos":True,
    "build_folders":True,
    "show_file_structure_example":True,
    "show_file_structure_length":False,
    "show_simu_run_example":False,
    "run_simu_example":False,
    "show_mean_results":False,
    "jsonify_synthesis_plot_data":False
}


# list of all variations of parameters studied
studies = [
    'LRW_CSMA_CADreliable', 
    'LRW_CSMA_PLE_ED', 
    'LRW_CSMA_PLE_GW', 
    'LRW_CSMA_capture', 
    'LRW_CSMA_fading', 
    'LRW_CSMA_interferers', 
    'LRW_CSMA_nbdev', 
    'LRW_CSMA_obstruction', 
    'LRW_CSMA_payload', 
    'LRW_CSMA_retries', 
    'LRW_CSMA_scale', 
    'LRW_CSMA_traffic'
]


if actions["build_topos"]:
    print("",file=sys.stderr)
    print("build_topos",file=sys.stderr)
    for stu in studies:
        topos={}

        max_nb_devs = 2000 if stu == "LRW_CSMA_nbdev" else 1000
        for topo_id in range(4):
            topos[topo_id]=topo_builder.build_topo(max_nb_devs,3,4,2000)

        pickle.dump(topos, open('results/{0}_topos.dat'.format(stu), 'wb'))

if actions["build_folders"]:
    print("",file=sys.stderr)
    print("build_folders",file=sys.stderr)
    for stu in studies:
        if not os.path.exists('results/{0}'.format(stu)):
            os.makedirs('results/{0}'.format(stu))

configs = pickle.load(open('LoRaWAN_CSMA_Improvements/LRW_CSMA_simu_runs.dat', 'rb'))

if actions["show_file_structure_example"]:
    print("",file=sys.stderr)
    print("show_file_structure_example",file=sys.stderr)
    run_id = 0
    for varparam in range(12):
        for varvalue in range(4):
            for topo in range(4):
                for proto_id in [0,1,2,3,4,5]:
                    for variant in range(64 if proto_id<5 else 1):
                        if variant == 0:
                            # print(run_id, configs[run_id]['run_index'], list(configs[run_id]['node_profiles']['clustered']['distrib'].keys())[0], file=sys.stderr)
                            if varparam==1 and proto_id==1:
                                print(configs[run_id]['gamma_ED'], configs[run_id]['topo'], file=sys.stderr)
                        run_id += 1


if actions["show_file_structure_length"]:
    print("",file=sys.stderr)
    print("show_file_structure_length",file=sys.stderr)
    print(len(configs), file=sys.stderr)

if actions["show_simu_run_example"]:
    print("",file=sys.stderr)
    print("show_simu_run_example",file=sys.stderr)
    for k in configs[13311].keys():
        print("{0}: {1}".format(k, configs[13311][k]),file=sys.stderr)

if actions["run_simu_example"]:
    print("",file=sys.stderr)
    print("run_simu_example",file=sys.stderr)

    simu.main_with_params(configs[13311])
    if actions["show_mean_results"]:
        print("",file=sys.stderr)
        print("show_mean_results",file=sys.stderr)

        res_dic=pickle.load(open('results/{0}/{0}_{1}_data.dat'.format(configs[13311]["start_time"],configs[13311]["run_index"]), 'rb'))

        # for a in res_dic:
        #   print(a, res_dic[a].keys())

        # print(res_dic["nodes"][66])
        # print(res_dic["settings"])
        for a in res_dic["TOTAL"]:
            print("{0}: {1}".format(a, res_dic["TOTAL"][a]),file=sys.stderr)


if actions["jsonify_synthesis_plot_data"]:
    print("",file=sys.stderr)
    print("jsonify_synthesis_plot_data",file=sys.stderr)

    figs=pickle.load(open('LoRaWAN_CSMA_Improvements/LRW_CSMA_simu_synthesis.dat', 'rb'))
    
    print(figs[96]["x_axis_label"],file=sys.stderr)
    print(figs[96]["y_axis_label"],file=sys.stderr)
    print(figs[96]["x_ticks"]["major"]["labels"],file=sys.stderr)
    print(figs[96]["plot_functions"][19]["args"][1][0],file=sys.stderr)


    json_dict = json.dumps(figs, indent=4)
    with open('LoRaWAN_CSMA_Improvements/LRW_CSMA_simu_synthesis.json', "w") as outfile:
        outfile.write(json_dict)