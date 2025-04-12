# -*- coding: utf-8 -*-
######################### Setup for studying CSMA protocols with LoRa-CSMA-Sim##############################################
######################### https://github.com/Guillaumegaillard/LoRa-CSMA-Sim ###############################################
######################### 2025-04-10 #######################################################################################
######################### License MIT: #####################################################################################
######################### https://github.com/Guillaumegaillard/LoRa-CSMA-Sim/blob/main/LICENSE #############################
######################### Inspired from LoRaSim https://mcbor.github.io/lorasim/ ###########################################

import pickle
import lora_csma_sim as simu
import os
import sys

if not os.path.exists('results/CSMA_comparison'):
    os.makedirs('results/CSMA_comparison')

aloha_run_file = "results/ALOHA_run.dat"
CSMA_run_file = "results/CSMA_comparison_run.dat"

# run_topos_file = "results/CSMA_comparison_topos.dat"
# run_topos = pickle.load(open(run_topos_file, 'rb'))
# print(len(run_topos))

aloha_run = pickle.load(open(aloha_run_file, 'rb'))
CSMA_run = pickle.load(open(CSMA_run_file, 'rb'))

for k in CSMA_run[13311].keys():
    print(k, CSMA_run[13311][k],file=sys.stderr)

# simu.main_with_params(CSMA_run[13311])
# simu.main_with_params(aloha_run[0])