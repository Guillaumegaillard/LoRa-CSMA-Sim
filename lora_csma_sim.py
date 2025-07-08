# -*- coding: utf-8 -*-
######################### Core module of the LoRa-CSMA-Sim Simulator #######################################################
######################### https://github.com/Guillaumegaillard/LoRa-CSMA-Sim ###############################################
######################### 2025-04-10 #######################################################################################
######################### License MIT: #####################################################################################
######################### https://github.com/Guillaumegaillard/LoRa-CSMA-Sim/blob/main/LICENSE #############################
######################### Inspired from LoRaSim https://mcbor.github.io/lorasim/ ###########################################

import simpy
import numpy as np
import math
import sys
import os
import pickle
import logging
from scipy.signal import savgol_filter


import constants

if not os.path.exists('results'):
    os.makedirs('results')

rng = np.random.default_rng()




#############################################
# Simulation Core Technical Things #
#############################################

#turn on/off print
#disabling printing to stdout will make simulation much faster
print_sim = True
print_sim = False
stdout_print_target=sys.stdout


#disable printing to stdout to make simulation much faster
if print_sim==False:
    f = open('/dev/null', 'w')
    sys.stdout = f    

# sys.stdout=stdout_print_target



################################
##############
# Local Constants      #
##############


#packet type
dataPacketType=1
rtsPacketType=2

#node's states to control the CANL22 mechanism
schedule_tx=0
want_transmit=1

CANL_listen1=10
CANL_NAV_state=11
CANL_send_RTS=12
CANL_listen2=13
CANL_send_DATA=14
CANL_CAD=15

#node's states to control the ideal mechanism
Ideal_send_DATA=16

#node type, you can add others and customize the behavior on configuration
endDeviceType=1
relayDeviceType=2

#packet transmit interval distribution type
expoDistribType=1
uniformDistribType=2
perioDistribType=3


################################
##### FUNCTIONS
################################

### If log_events is acitvated, the call to this function will initiate the logs process
def load_main_logger(start_time):
    
    ###########LOGGING MAIN EVENTS
    MainLogger = logging.getLogger('Main_Logger')
    MainLogger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    mfh = logging.FileHandler('{0}.log'.format(start_time), mode='w')
    mfh.setLevel(logging.INFO)
    #console
    mch = logging.StreamHandler()
    mch.setLevel(logging.CRITICAL)
    # create formatter and add it to the handlers
    mformatter = logging.Formatter('%(asctime)s %(message)s')
    mfh.setFormatter(mformatter)
    mch.setFormatter(mformatter)
    
    close_logger_handlers(MainLogger)
    
    # add the handlers to logger
    MainLogger.addHandler(mfh)
    MainLogger.addHandler(mch)
    
    return (MainLogger)

### shut down logging when experiment stops
def close_logger_handlers(logger_obj):
    handlers = logger_obj.handlers[:]
    for handler in handlers:
        handler.close()
        logger_obj.removeHandler(handler)



############################
############################
# check for receptions/collisions/captures at base station(s) and among devices
# Note: called before a packet (identified with node id) is inserted into the list
# takes as parameter "heard_at_gateways", an array of booleans indicating detection at each GW
def checkcollision(packet,heard_at_gateways):
    global nrCaptures
    col = 0 # flag needed since there might be several collisions for packet

    # if packet is detected at at least one GW
    if not packet.lost:
        ### packet.processed is not used in this version
        processing = 0
        for i in range(0,len(packetsOnAir)):
            if nodes[packetsOnAir[i]].packet.processed == 1 :
                processing = processing + 1
        if (processing > maxBSReceives):
            # print("too many packets on air:", len(packetsOnAir))
            packet.processed = 0
        else:
            packet.processed = 1


        # if somebody else is transmitting
        if len(packetsOnAir)>0:
            # count detections and signal mWs at GWs
            ## count me
            in_ears_GWs=np.where(heard_at_gateways,1,0) # packet is not yet in packetsOnAir but will be part of the mess
            this_interf_mw_GWs = 10**(.1*packet.rss_GWs)

            # count all others
            for other in packetsOnAir:
                if frequencyCollision(packet, nodes[other].packet): # hyp: channels are orthogonal
                    other_heard_at_gateways = (nodes[other].packet.rss_GWs - nodes[other].packet.noise_dB  > get_sensitivity(nodes[other].packet.sf,nodes[other].packet.bw))
                    in_ears_GWs+=np.where(other_heard_at_gateways,1,0)

                    other_packet_signal_at_gws_mw = 10**(.1*nodes[other].packet.rss_GWs)

                    # add interference to current packet
                    packet.interf_mw_GWs+=other_packet_signal_at_gws_mw
                    # add interference to other packet
                    nodes[other].packet.interf_mw_GWs += this_interf_mw_GWs


            #### EVALUATE COLLISIONS AND CAPTURES 
            for other in packetsOnAir:
                for gw_id in range(len(heard_at_gateways)): # for each gw
                    if heard_at_gateways[gw_id]:
                        #check if this other packet is (different than packet and) heard at GW
                        if nodes[other].nodeid != packet.nodeid and (nodes[other].packet.rss_GWs[gw_id] - nodes[other].packet.noise_dB[gw_id] > get_sensitivity(nodes[other].packet.sf,nodes[other].packet.bw)):
                            # check if same sf, same channel (simple collision)
                            if frequencyCollision(packet, nodes[other].packet) and sfCollision(packet, nodes[other].packet):
                                if full_collision: # (this global is always true since 2022...)
                                    # check who collides in the power domain, considering interference and preamble lock
                                    c = powerCollision(packet, nodes[other].packet, in_ears=in_ears_GWs[gw_id], ptype="gw", loc_id=gw_id)
                                    # either this one, the other one, or both

                                    receiving_packet = False
                                    # mark all the collided packets
                                    if packet in c: # packet may be spared by preamble in a few symbols
                                        # packet collides in power, check if the other finishes before most of preamble 
                                        clear_pream_time_heard_here, not_enough_preamble = timingCollision(packet, nodes[other].packet)
                                        if not_enough_preamble:# packet won't be rec there
                                            col = 1
                                            packet.collided_at_GWs[gw_id] = 1
                                            nodes[packet.nodeid].n_captured +=1
                                            nrCaptures+=1

                                            if log_events:
                                                MainLogger.info(("GW{0}".format(gw_id),"col",packet.nodeid,packet.nodeid,env.now))
                                        else:
                                            receiving_packet = True
                                    else:
                                        receiving_packet = True

                                    # lock receptor on packet
                                    if receiving_packet:
                                        packet.receiving_GWs[gw_id]=True
                                        packet.receiving_start_GWs[gw_id]=env.now

                                    # mark older packets as collided if needed
                                    if nodes[other].packet in c:
                                        if nodes[other].packet.collided_at_GWs[gw_id] == 0:
                                            nodes[other].n_captured +=1
                                            nodes[other].packet.collided_at_GWs[gw_id] = 1
                                            nrCaptures+=1
                                        if log_events:
                                            MainLogger.info(("GW","col",packet.nodeid,nodes[other].packet.nodeid,env.now))
                                        nodes[other].packet.receiving_GWs[gw_id]=False
                                else: # both collide
                                    nodes[packet.nodeid].n_captured +=1
                                    packet.collided_at_GWs[gw_id] = 1
                                    nrCaptures+=1
                                    if nodes[other].packet.collided_at_GWs[gw_id] == 0:
                                        nodes[other].n_captured +=1
                                        nodes[other].packet.collided_at_GWs[gw_id] = 1     # other also got collided, if it wasn't collided already
                                        nrCaptures+=1
                                    if log_events:
                                        MainLogger.info(("GW{0}".format(gw_id),"col",packet.nodeid,packet.nodeid,env.now))
                                        MainLogger.info(("GW{0}".format(gw_id),"col",packet.nodeid,nodes[other].packet.nodeid,env.now))
                                    col = 1
            

    ############### Packet has been received at GW. Let's see elsewhere
    if not full_distances:    
        #if col==1 it means that the new packet can not be decoded
        if col:
            return col    
            
    #normally, here, the packet has been correctly received at GW (not collided)
    if not col:
        if log_events:
            MainLogger.info(("GW","rx",packet.nodeid,env.now))

    # old "everywhere is the same as GW" case 
    if not full_distances:   
        # not collided at GW, packet is alone on air 

        #the trick is to assume that if the gateway received a packet
        #then all other nodes in listening state should also have receive it
        for node in nodes:
            if node.nodeid != packet.nodeid:
                # if profiles[node.profile]["protocol"]=="CANL22":
                #node is listenning
                if node.MAC_state in [CANL_listen1, CANL_listen2]:

                    # mark the packet in list of receptions during listenning 
                    # depending on the scenario in terms of headers, the data payload size could be known, unknown or mistaken by the receiver
                    # dataPayloadSize_in_RTS is used only in a scenario with RTS 
                    # packet could be partially interfered by/interfering other packets on air - checked later wrt times & powers 
                    node.heard_frames.append({
                        "id":len(node.heard_frames),
                        "nodeid":packet.nodeid,
                        "toa":packet.rectime,
                        "is_RTS":packet.ptype == rtsPacketType,
                        "dataPayloadSize_in_RTS":rng.integers(0,max_payload_size+1) if packet.dataPayloadSize==(profiles[node.profile]["CANL22_RTS_hdr_size"]+1) else packet.dataPayloadSize, # if Data of RTS size, random size
                        "dataPayloadSize_in_EH":packet.dataPayloadSize, # in explicit header
                        "start":env.now,
                        "interfering":[(i,nodes[i].packet.ptype) for i in packetsOnAir],
                        "interfered_by":[(i,nodes[i].packet.ptype) for i in packetsOnAir],
                        })
                    if log_events:
                        MainLogger.info((node.nodeid,"rx",packet.nodeid,env.now))

    else:
        # # general case 
        # for each node listening
        #     if heard
        #         for other packets on air 
        #             if collide?Capture
        #                 adjust
        #         record impact/or not

        for node in nodes:
            # if CANL22:
            if profiles[node.profile]["protocol"]=="CANL22":
                locally_collided=False
                previous_frames_impacted_by_this_one=[]
                previous_frames_impacting_this_one=[]
                clear_pream_time_heard = 10000
                if node.MAC_state in [CANL_listen1, CANL_listen2]:
                    if check_heard(packet,node.nodeid):
                        in_ears=1 # packet is in ears

                        # packet has those few mW of power strength at node.nodeid 
                        this_mw_this_dev = 10**(.1*(packet.rx_array[node.nodeid]))

                        for other in packetsOnAir:
                            # Only count mWs of perceived packets (TODO: also sum mW below sensitivity?)
                            if check_heard(nodes[other].packet,node.nodeid):
                                in_ears+=1

                                other_signal_mw = 10**(.1*(nodes[other].packet.rx_array[node.nodeid]))

                                packet.interf_mw[node.nodeid] += other_signal_mw

                                nodes[other].packet.interf_mw[node.nodeid] += this_mw_this_dev

                        # study collisions//capture on node.nodeid
                        for other in packetsOnAir:
                            if check_heard(nodes[other].packet,node.nodeid):
                                if frequencyCollision(packet, nodes[other].packet) and sfCollision(packet, nodes[other].packet):
                                    if full_collision:
                                        # check who collides in the power domain
                                        c = powerCollision(packet, nodes[other].packet, in_ears=in_ears,loc_id=node.nodeid)
                                        # either this one, the other one, or both

                                        #####################################################
                                        #####################################################

                                        # check if node could focus on packet's preamble to "lock" its reception on it
                                        receiving_packet = False
                                        # mark all the collided packets
                                        if packet in c:
                                            clear_pream_time_heard_here, not_enough_preamble = timingCollision(packet, nodes[other].packet)
                                            clear_pream_time_heard = min(clear_pream_time_heard, clear_pream_time_heard_here)
                                            if not_enough_preamble:# both_collide, or just the other?
                                                locally_collided=True
                                                previous_frames_impacting_this_one.append((nodes[other].packet.nodeid,nodes[other].packet.ptype))
                                                if log_events:
                                                    MainLogger.info((node.nodeid,"col",packet.nodeid,packet.nodeid,env.now))
                                            else:
                                                receiving_packet = True
                                        else:
                                            receiving_packet = True

                                        if receiving_packet: # in good condition to lock on packet when enough preamble symbols heard (but another frame may occur in future and impair that)
                                            packet.receiving[node.nodeid]=True
                                            packet.receiving_start[node.nodeid]=env.now

                                        if nodes[other].packet in c:
                                            if log_events:
                                                MainLogger.info((node.nodeid,"col",packet.nodeid,nodes[other].packet.nodeid,env.now))
                                            previous_frames_impacted_by_this_one.append((nodes[other].packet.nodeid,nodes[other].packet.ptype))
                                            nodes[other].packet.receiving[node.nodeid]=False
                                    else:
                                        if log_events:
                                            MainLogger.info((node.nodeid,"col",packet.nodeid,nodes[other].packet.nodeid,env.now))
                                            MainLogger.info((node.nodeid,"col",packet.nodeid,packet.nodeid,env.now))
                                        previous_frames_impacting_this_one.append((nodes[other].packet.nodeid,nodes[other].packet.ptype))
                                        previous_frames_impacted_by_this_one.append((nodes[other].packet.nodeid,nodes[other].packet.ptype))
                                        locally_collided=True

                        # see above, mark the packet in list of receptions during listenning 
                        node.heard_frames.append({
                            "id":len(node.heard_frames),
                            "nodeid":packet.nodeid,
                            "toa":packet.rectime,
                            "is_RTS":packet.ptype == rtsPacketType,
                            "dataPayloadSize_in_RTS":rng.integers(0,max_payload_size+1) if packet.dataPayloadSize==(profiles[node.profile]["CANL22_RTS_hdr_size"]+1) else packet.dataPayloadSize, # if Data of RTS size, random size
                            "dataPayloadSize_in_EH":packet.dataPayloadSize, # in explicit header
                            "start":env.now,
                            "interfering":previous_frames_impacted_by_this_one,
                            "interfered_by":previous_frames_impacting_this_one,
                            "clear_pream_time_heard":clear_pream_time_heard
                        })

                        ######### packet is detected, stop normal listen process and keep listening the time necessary for a valid header to be received 
                        try:
                            node.listen_process.interrupt()
                        except RuntimeError:
                            # node has just stopped listenning, listen_process is already interrupted
                            # should not occur anymore (2025-04-10)
                            pass

                        
                        if not locally_collided:
                            if log_events:
                                MainLogger.info((node.nodeid,"rx",packet.nodeid,env.now))

    return col



# 
# check if a device perceives a neighbor frame above its sensitivity threshold 
# (on the channel its tuned on)
def check_heard(packet,nodeid):
    rss=packet.rx_array[nodeid]
    if frequencyCollision(packet, nodes[nodeid].packet):
        return (rss - packet.noise_dB_arr[nodeid] >= get_sensitivity(packet.sf,packet.bw,receiver_type="DEVICE"))

    return False

#
# retrieve sensitivity according to scenario and dev type
# watch out: two elements are compared:
### 1: sensitivity as given in Semtech specs, i.e. the minimal signal level detected accounting for:
########## a JN noise floor, depending on the bandwidth
########## a typical noise figure of 6dB accounting for noise in equipments (HW noise)  
### 2: effective signal reception:
########## affected on its path by shadowing and path loss;
########## affected locally by a lognormal noise figure;
###
###
### => in order to compare meaningfull values, constants.noise_floor_ST should be substracted from both elements
### in this case actual SNRs would be compared with actual SNR thresholds

def get_sensitivity(spreading_factor,bandwidth,receiver_type="GW"):
    if experiment == 4:
        if receiver_type=="DEVICE":
            return(constants.sensi_subGHz[spreading_factor - 6, [125,250,500].index(bandwidth) + 1]) 
        return(-GW_sensitivity_gain+constants.sensi_GW_boosted[bandwidth][spreading_factor]) # -137 - 1.5
    if experiment == "SF7BW500CAD4":
        if receiver_type=="DEVICE":
            return(constants.sensi_subGHz[spreading_factor - 6, [125,250,500].index(bandwidth) + 1]) 
        return(-127)
        # return(-150)    
    if lora24GHz:
        sensitivity = constants.sensi_2dot4GHz[spreading_factor - 5, [203.125,406.25,812.5,1625].index(bandwidth) + 1]
    else:
        sensitivity = constants.sensi_subGHz[spreading_factor - 6, [125,250,500].index(bandwidth) + 1]

    return(sensitivity)


# # OLD frequencyCollision in a scenario with overlapping channels 
# # frequencyCollision, conditions
# #
# #         |f1-f2| <= 120 kHz if f1 or f2 has bw 500
# #         |f1-f2| <= 60 kHz if f1 or f2 has bw 250
# #         |f1-f2| <= 30 kHz if f1 or f2 has bw 125
# def old_frequencyCollision(p1,p2):
#     if (abs(p1.freq-p2.freq)<=120 and (p1.bw==500 or p2.bw==500)):
#         return True
#     elif (abs(p1.freq-p2.freq)<=60 and (p1.bw==250 or p2.bw==250)):
#         return True
#     else:
#         if (abs(p1.freq-p2.freq)<=30):
#             return True
#     # print("no frequency coll")
#     return False


#
# Check if two packet are transmitted on same channel
def frequencyCollision(p1, p2):
    if nodes[p1.nodeid].ch == nodes[p2.nodeid].ch:
        return True
    return False


#
# Check if two packet are transmitted with same spreading factor
def sfCollision(p1, p2):
    if p1.sf == p2.sf:
        return True
    return False

#
# Check and store the reception conditions of two packets 
# at a given device (loc_id=devID) // at a GW (loc_id=gw_ID)
# depends on how many transmissions are perceived at receiver at the time
# depends on how much interference there is
# depends whether the receiver is tuned ("locked", "synced") on a signal or not
def powerCollision(p1, p2, in_ears=2,loc_id=-1, ptype="dev", ocurring_now=True):

    # mWs have been collected earlier
    if ptype!="gw":
        if p1.interf_mw[loc_id]>0: 
            SIR_1 = p1.rx_array[loc_id] - 10*np.log10(p1.interf_mw[loc_id])
        else:
            SIR_1 = 200 # default max

        if p2.interf_mw[loc_id]>0:
            SIR_2 = p2.rx_array[loc_id] - 10*np.log10(p2.interf_mw[loc_id])
        else:
            SIR_2 = 200

        # count preamble symbs of old P2
        p2synced = False
        if p2.receiving[loc_id]:
            p1a = env.now if ocurring_now else p1.addTime
            if p1a - p2.receiving_start[loc_id] >= p2.Tmin_preamb_heard : #Rx has synced on old (enough symbs)
                p2synced = True

        # when the receiver is "locked", it is less impacted by interference
        interf_coef = 1
        if p2synced and SIR_2 > 0:
            interf_coef = capture_interf_lock_coef # global parameter to be adjusted for each run (e.g. .5)

        # check receptions, considering a non linear impact of interference
        p1_rec = p1.rx_array[loc_id] - p1.noise_dB_arr[loc_id] + constants.interf_impact_sensi(SIR_1) >= get_sensitivity(p1.sf,p1.bw,receiver_type="DEVICE")
        p2_rec = p2.rx_array[loc_id] - p2.noise_dB_arr[loc_id] + interf_coef*constants.interf_impact_sensi(SIR_2) >= get_sensitivity(p2.sf,p2.bw,receiver_type="DEVICE")

    else: # same on a GW
        if p1.interf_mw_GWs[loc_id]>0:
            SIR_1 = p1.rss_GWs[loc_id] - 10*np.log10(p1.interf_mw_GWs[loc_id])
        else:
            SIR_1 = 200
        if p2.interf_mw_GWs[loc_id]>0:
            SIR_2 = p2.rss_GWs[loc_id] - 10*np.log10(p2.interf_mw_GWs[loc_id])
        else:
            SIR_2 = 200

        p2synced = False
        if p2.receiving_GWs[loc_id]:
            p1a = env.now if ocurring_now else p1.addTime
            if p1a - p2.receiving_start_GWs[loc_id] >= p2.Tmin_preamb_heard : #Rx has synced on old (enough symbs)
                p2synced = True

        interf_coef = 1
        if p2synced and SIR_2 > 0:
            interf_coef = capture_interf_lock_coef

        p1_rec = p1.rss_GWs[loc_id] - p1.noise_dB[loc_id] + constants.interf_impact_sensi(SIR_1) >= get_sensitivity(p1.sf,p1.bw)
        p2_rec = p2.rss_GWs[loc_id] - p2.noise_dB[loc_id] + interf_coef*constants.interf_impact_sensi(SIR_2) >= get_sensitivity(p2.sf,p2.bw)

    
    ## Conclude: did p1 break P2's reception? was p1 hidden by p2?
    if (not p1_rec):
        if (not p2_rec):
            powerCaptures.append((in_ears,False,loc_id,ptype))
            return (p1, p2)
        powerCaptures.append((in_ears,True,loc_id,ptype))
        return (p1,)
    # if p1_rec then p2 is not rec (both cannot be rec simultaneously (on devices... on GW? => TODO))
    powerCaptures.append((in_ears,True,loc_id,ptype))
    return (p2,)        



#
# check the time concommitance of two transmissions
def timingCollision(p1, p2, ocurring_now=True):
    # assuming p1 is the freshly arrived packet and this is the last collision check
    # we know p2 is not finished (it would not be on air, if the contrary was true)
    # if we entered here, we know that p2 is in time collision

    # we've already determined that p1 is a weak packet, so the only
    # way we can win is by being late enough (only the first n - x preamble symbols overlap)
    # x being detection_min_preamb_symb
    
    # check whether p2 ends in p1's critical section
    p2_end = p2.addTime + p2.rectime
    p1_cs_start = env.now + p1.Tpream - p1.Tmin_preamb_heard if ocurring_now else p1.addTime + p1.Tpream - p1.Tmin_preamb_heard 

    # free chunk of p1's preamble
    preamb_heard = env.now + p1.Tpream - p2_end if ocurring_now else p1.addTime + p1.Tpream - p2_end

    if p1_cs_start < p2_end:
        # p1 collided with p2 and lost
        # print("not late enough")
        return (preamb_heard, True)
    # print("spared by the preamble")
    return (preamb_heard, False)


#
# this function computes the airtime of a packet
# according to LoraDesignGuide_STD.pdf
def airtime(sf,cr,pl,bw,explicit=True):
    
    DE = 0         # low data rate optimization enabled (=1) or not (=0)
    Npream = 8     # number of preamble symbol (12.25     from Utz paper)

    if explicit: # Header
        H=0
    else:
        H=1

    if lora24GHz:
        Npream = 12
        H = 1         # header for variable length packet (H=1) or not (H=0)        
        if sf > 10:
            # low data rate optimization mandated for SF > 10
            DE = 1
        Tsym = (2.0**sf)/bw
        if sf < 7:
            Tpream = (Npream + 6.25)*Tsym
        else:
            Tpream = (Npream + 4.25)*Tsym
        #print("sf", sf, " cr", cr, "pl", pl, "bw", bw
        if sf >= 7:
            payloadSymbNB = 8 + math.ceil(max((8.0*pl+16-4.0*sf+8+20*H),0)/(4.0*(sf-2*DE)))*(cr+4)
        else:
            payloadSymbNB = 8 + math.ceil(max((8.0*pl+16-4.0*sf+20*H),0)/(4.0*(sf-2*DE)))*(cr+4)
        Tpayload = payloadSymbNB * Tsym
        return (Tpream + Tpayload)     
    else:
        if bw == 125 and sf in [11, 12]:
            # low data rate optimization mandated for BW125 with SF11 and SF12
            DE = 1
        if sf == 6:
            # can only have implicit header with SF6
            H = 0
        Tsym = (2.0**sf)/bw
        Tpream = (Npream + 4.25)*Tsym
        #print("sf", sf, " cr", cr, "pl", pl, "bw", bw
        # CR is 1, 2, 3 or 4 for respective coding rates 4/5, 4/6, 4/7 or 4/8
        payloadSymbNB = 8 + max(math.ceil((8.0*pl-4.0*sf+28+16-20*H)/(4.0*(sf-2*DE)))*(cr+4),0)
        Tpayload = payloadSymbNB * Tsym
        return(Tpream + Tpayload)
    
#
# the node (device) object definition
#
class myNode():
    # a node has an id, coordinates, a type (relay, endDevice...), a cluster, a profile, a traffic period, distribution, and default payload size
    def __init__(self, nodeid, nodex, nodey, node_cluster, node_profile, nodeType, period, distrib, packetlen):
        self.nodeid = nodeid
        self.x = nodex
        self.y = nodey
        self.cluster=node_cluster
        self.profile=node_profile       # node profile
        self.nodeType = nodeType        # relay, enddev... not used currently
        self.period = period            # average generate (send) time period
        self.distrib = distrib          # generate (send) prob distrib

        # distance to center of topology
        self.dist_to_center = np.sqrt((self.x-center_x)*(self.x-center_x)+(self.y-center_y)*(self.y-center_y))
        
        # distance to GWs
        self.dist_GWs = np.sqrt((self.x-gateways["xs"])**2+(self.y-gateways["ys"])**2)

        # node has a packet
        self.packet = myPacket(self.nodeid, packetlen, self.dist_GWs)
        
        # packet has a time on air called rectime
        self.data_rectime = airtime(self.packet.sf,self.packet.cr,self.packet.pl,self.packet.bw,explicit=(LoRa_PHY_HDR==0))

        # available channels for next ch update
        self.avail_chans = [i for i in range(len(constants.Channel_list))]

        # channel is randomly initialized
        self.ch = rng.choice(self.avail_chans)
        self.channel_just_changed = True
        self.last_tx_ch = -1

        # channels tried and found busy
        self.failed_chans = []

        # time required for configuring the radio to a given channel
        self.radio_config_time = radio_config_time #5 #ms
        # time required to wake up the radio
        self.radio_wake_up_time = radio_wake_up_time #5 #ms

        # statistics
        self.radio_config_number = 0
        self.radio_wake_up_number = 0

        # last moment radio was on
        self.radio_on_last = -1000000

        # configure stuffs for CANL22
        if profiles[self.profile]["protocol"]=="CANL22":
            # configure a RTS packet for CANL22 (not in use in 2025)
            # if a RTS is used, its payload is 1 byte
            self.rts_rectime = airtime(self.packet.sf,self.packet.cr,profiles[self.profile]["CANL22_RTS_hdr_size"]+1,self.packet.bw,explicit=(profiles[self.profile]["CANL22_RTS_PHY_HDR"]==0))

            # time a device stays listening in order to obtain data length information
            if profiles[self.profile]["Interrupts_on_header_valid"]: # device is capable of using header right after the "header valid" interrupt
                # self.wait_PHY_interrupt = airtime(self.packet.sf,self.packet.cr,0,self.packet.bw,explicit=(LoRa_PHY_HDR==0))

                # CR is 1, 2, 3 or 4 for respective coding rates 4/5, 4/6, 4/7 or 4/8
                # header is tx with cr 4
                self.wait_PHY_interrupt = airtime(self.packet.sf,4,0,self.packet.bw,explicit=(LoRa_PHY_HDR==0))
            else:
                self.wait_PHY_interrupt = self.rts_rectime# + self.packet.symTime*nCadSym # I stay listening a margin to be sure to receive => not here, in listen
        else: # RTS is not used anyway, let's consider a 5B length for compatibility
            self.rts_rectime = airtime(self.packet.sf,self.packet.cr,5,self.packet.bw,explicit=(LoRa_PHY_HDR==0))
            self.wait_PHY_interrupt = airtime(self.packet.sf,self.packet.cr,5,self.packet.bw,explicit=(LoRa_PHY_HDR==0))



        ################## NODE VARIABLES ##########################
        self.n_data_sent = 0
        self.n_rts_sent = 0
        self.n_data_success = 0 # successfully transmitted frames
        self.n_payload_gen = 0 # amount of generated bytes
        self.n_payload_sent = 0 # amount of sent bytes
        self.n_payload_success = 0 # amount of received bytes

        self.want_transmit_time=0
        self.queued_time=0

        #### Variables for collision avoidance /listen
        self.MAC_state = schedule_tx # initial state, will change in communicate() function
        self.MAC_listen_start_time = 0
        self.MAC_listen_end_time = 0
        self.listened_time=-1
        self.total_listen_time = 0 # for all time in simu

        self.my_P = 0 # probability for an old CANL version (not in use 2025)
        self.backoff = 0 # time to be spent in BO, updated by protocols
        self.remainder = -1 # remaining steps to be done in a residual active BO

        ## variables for listenning          
        self.heard_frames=[] # list of heard frames and metadata
        self.add_rx_time_opportunities=1 # when a frame starts being received, how many times the listen window can be elongated.

        ## listen result variables
        self.I_heard_preamble=False
        self.I_heard_activity=False
        self.I_know_it_is_Data=False
        self.I_know_it_is_RTS=False
        self.next_payload_byte=0

        ## DEFAULT NAV PERIOD, which is the toa(max sized packet) - one CAD time
        self.nav_period_default = airtime(self.packet.sf,self.packet.cr,max_payload_size,self.packet.bw,explicit=(LoRa_PHY_HDR==0))-self.packet.symTime*(nCadSym+1/2)
        # variable initialized at default
        self.nav_period = self.nav_period_default

        self.nav=0 # variable indicating a required NAV
        self.n_CAD=0 # number of cads done 

        ### mechanism n_retry
        ## variable number of remaining possible attempts for the current packet 
        # init according to profile
        self.n_retry=profiles[self.profile]["n_retry"] if "n_retry" in profiles[self.profile] else -1
        self.total_retry=0 # stat for whole sim

        ## stats for node during whole simulation                    
        self.n_aborted=0
        self.n_collided=0
        self.n_captured = 0
        self.n_lost=0
        self.n_dropped=0
        self.cycle=0
        self.kamikaze=0

        ## latencies
        self.latency=0 # of current packet 
        self.success_latency=0 # of all packets, sumed and then averaged
        self.min_success_latency=self.period*1000 # init at very large value

        ## BO exponent for BO when BUSY (init at value in profile)
        if "Wbusy_BE" in profiles[self.profile]:
            self.Wbusy_BE=profiles[self.profile]["Wbusy_BE"]

        ### variables for packet generation and queue
        self.gen_times=[0]
        self.queue = []
        self.next_gen_time=-1#ms


    # determine how much extra time needed to detect preamble, if possible
    def time_to_detect_preamble(self):
        nec_time = -1

        lhf=len(self.heard_frames)
        if lhf==0:#I heard nothing, go back listening
            pass 
        else:
            for frame_heard_id in range(lhf):# in chrono order
                frame_heard=self.heard_frames[frame_heard_id]

                # was it captured from the beginning?
                if len(frame_heard["interfered_by"])==0: # no
                    #did I hear it start? (was i listening, is preamble finished?)
                    # yes, => wait for more info / stop listening
                    # no, => wait for more info / stop listening

                    # I am listening and the preamble had not yet finished
                    if (env.now<frame_heard["start"]+self.packet.Tpream-self.packet.Tmin_preamb_heard):
                        
                        # two cases: no frame before//frame before
                        # frame before
                        if frame_heard["clear_pream_time_heard"]<self.packet.Tpream:
                            start_hearing_it = max(
                                frame_heard["start"]+self.packet.Tpream-frame_heard["clear_pream_time_heard"],
                                env.now
                                )
                        else:
                            start_hearing_it = max(
                                frame_heard["start"],
                                env.now
                                )



                        # was it captured after its beginning (and before now)? 
                        first_capturer=-1
                        for nextid in range(frame_heard_id+1,lhf): # last does not enter here
                            if (frame_heard["nodeid"], rtsPacketType if frame_heard["is_RTS"] else dataPacketType) in self.heard_frames[nextid]["interfering"]:
                                first_capturer=nextid
                                break
                        if first_capturer!=-1: # yes
                            # Then I don't hear it, no matter what...
                            pass
                        else:
                            # (will be) hearing enough preamble symbols if not interrupted 
                            return (frame_heard["id"],(start_hearing_it+self.packet.Tmin_preamb_heard-env.now))


        return (-1,nec_time)


    # determine how much extra time needed to detect full (valid) header, if possible
    def time_to_wait_vh(self, start_listening, frame_id):

        lhf=len(self.heard_frames)
    
        for frame_heard_id in range(lhf):# in chrono order
            frame_heard=self.heard_frames[frame_heard_id]
            if frame_heard["id"] == frame_id:
                break

        # two cases: no frame before//frame before
        # frame before
        if frame_heard["clear_pream_time_heard"]<self.packet.Tpream:
            started_hearing_it = max(
                frame_heard["start"]+self.packet.Tpream-frame_heard["clear_pream_time_heard"],
                start_listening
            )
        else:
            started_hearing_it = max(
                frame_heard["start"],
                start_listening
            )

        return((frame_heard["start"]+self.wait_PHY_interrupt - env.now),(started_hearing_it+self.wait_PHY_interrupt+self.packet.symTime - env.now)) # margin one symbol

    # determine if frame has been impacted by later frames
    def check_frame_captured(self, frame_id):
        lhf=len(self.heard_frames)
    
        for frame_heard_id in range(lhf):# in chrono order
            frame_heard=self.heard_frames[frame_heard_id]
            if frame_heard["id"] == frame_id:
                break

        # was it captured after its beginning (and before now)? 
        first_capturer=-1
        for nextid in range(frame_heard_id+1,lhf): # 
            if (frame_heard["nodeid"], rtsPacketType if frame_heard["is_RTS"] else dataPacketType) in self.heard_frames[nextid]["interfering"]:
            # if frame_heard["nodeid"] in self.heard_frames[nextid]["interfering"]:
                first_capturer=nextid
                break

        return(first_capturer!=-1)              


    # grab info from header
    def read_header(self, frame_id):

        lhf=len(self.heard_frames)
    
        for frame_heard_id in range(lhf):# in chrono order
            frame_heard=self.heard_frames[frame_heard_id]
            if frame_heard["id"] == frame_id:
                break

        if frame_heard["toa"]>self.wait_PHY_interrupt: # I know it has payload
            self.I_know_it_is_Data=True
            # assert(frame_heard["is_RTS"]==False)
            if profiles[self.profile]["Interrupts_on_header_valid"]:
                self.next_payload_byte=frame_heard["dataPayloadSize_in_EH"]
        else: # I know it has no payload
            if frame_heard["is_RTS"]: #I know it by CANL header differentiation (data, RTS, ACK) 
                self.I_know_it_is_RTS=True
                self.next_payload_byte=frame_heard["dataPayloadSize_in_RTS"]
            else: # short data
                self.I_know_it_is_Data=True
                if profiles[self.profile]["Interrupts_on_header_valid"]:
                    self.next_payload_byte=frame_heard["dataPayloadSize_in_EH"]

    # listen process: can be initiated by protocol or prolongated if necessary
    def listen(self, duration):
        start_listening = env.now

        # check if a rx has started (otherwise detecting_preamble == -1)
        frame_id, detecting_preamble = self.time_to_detect_preamble()

        # prolongate if permitted
        while self.add_rx_time_opportunities>=0 and detecting_preamble>0:
            # listen full preamble or catch a new heard rx 
            try:
                yield env.timeout(detecting_preamble)
                self.I_heard_preamble=True                
                break
            except simpy.Interrupt:
                # if current captured during preamble, new packet can be heard (e.g. if strong enough)
                # recheck detection
                frame_id, detecting_preamble = self.time_to_detect_preamble()
                # if later detected, remove one opportunity 
                if detecting_preamble != -1 and (env.now + detecting_preamble - start_listening > duration):
                    self.add_rx_time_opportunities-=1

        # same idea, once preamble heard, wait for header
        if self.I_heard_preamble:

            #wait vh
            waiting_for_vh,waiting_default = self.time_to_wait_vh(start_listening, frame_id)
            start_waiting_for_vh = env.now

            heard_vh = True # will get False if capture
            time_waiting = waiting_for_vh
            while True:

                # whether wait or be interrupted
                try:
                    yield env.timeout(time_waiting)
                    break
                except simpy.Interrupt:
                    captured = self.check_frame_captured(frame_id)
                    if captured: # I won't hear valid header, so I wait default (full preamble)
                        heard_vh = False
                        time_waiting = waiting_default - (env.now - start_waiting_for_vh)
                    else: # I can hear end of header, and stop listening 
                        time_waiting = waiting_for_vh - (env.now - start_waiting_for_vh)

            # if header heard, update listener's info
            if heard_vh:
                self.read_header(frame_id)
            else:
                pass
            return(-1)

        else:
            #default listening
            start_default_listen = env.now
            remaining_to_listen = duration - (start_default_listen - start_listening)
            while remaining_to_listen>0:
                # wait or be interrupted
                try:
                    yield env.timeout(remaining_to_listen)
                    break
                except simpy.Interrupt: # something was heard, break, and re_entering, decide if resume or stop listening
                    remaining_to_listen -= (env.now - start_default_listen)
                    return(remaining_to_listen)
            return(-1)




    # node function to update the channel to which radio is tuned
    # does not yield any delay here (see main "communicate" loop)
    # return False if no new channel is available
    def update_channel(self):
        self.failed_chans.append(self.ch)

        non_failed_chans = [i for i in self.avail_chans if (i not in self.failed_chans)]
        if len(non_failed_chans)>0:
            self.ch = rng.choice(non_failed_chans)
            return(True)
        else:
            return(False)


    # node function to update/reset the list of available channels
    def update_available_channels(self):
        if len(self.avail_chans) == 1:
            self.avail_chans = [i for i in range(len(constants.Channel_list))]
        if len(self.avail_chans) > 1:
            # self.avail_chans = [i for i in self.avail_chans if i!=self.ch]
            self.avail_chans = [i for i in self.avail_chans if i!=self.last_tx_ch]

        self.failed_chans = []


        
    # node function called (in CANL) at beginning of listen phase
    # Check if packets on air are heard and impacting the node
    def start_listening(self):  

        # for each packet on air oldest first 
        #     was it heard?
        #     has it been captured before even its begin?
        #     does it capture previous?

        if len(packetsOnAir)>0:
            # count current mWs of interference and current interferers 
            in_ears=0
            full_mw_devs = 0
            for other in packetsOnAir:
                if check_heard(nodes[other].packet,self.nodeid):
                    in_ears+=1
                    full_mw_devs += 10**(.1*(nodes[other].packet.rx_array[self.nodeid]))

            # interference suffered by a packet p is the full sum of strengthes received minus p's mWs 
            for other in packetsOnAir:
                if check_heard(nodes[other].packet,self.nodeid):
                    nodes[other].packet.interf_mw[self.nodeid] = full_mw_devs - 10**(.1*(nodes[other].packet.rx_array[self.nodeid]))

            ## check collisions/captures, in chrono order and chrono order of preceding
            for pid in range(len(packetsOnAir)):
                packet=nodes[packetsOnAir[pid]].packet
                if check_heard(packet,self.nodeid): #NB: check_heard checks frequency channel
                    previous_frames_impacting_this_one=[]
                    previous_frames_impacted_by_this_one=[]   
                    clear_pream_time_heard = 10000   
                    locally_collided=False                  
                    if pid!=0: # more than one on air
                        for opid in range(pid):
                            other=packetsOnAir[opid]
                            if check_heard(nodes[other].packet,self.nodeid):
                                if frequencyCollision(packet, nodes[other].packet) and sfCollision(packet, nodes[other].packet): #TODO: frequency has already been checked, sfCollision is sufficient
                                    if full_collision:
                                        # check who collides/captures/interfers in the power domain
                                        # account for interf impact and lock of rx
                                        c = powerCollision(packet, nodes[other].packet,in_ears=in_ears,loc_id=self.nodeid, ocurring_now=False)
                                        # either this one, the other one, or both

                                        receiving_packet = False
                                        # mark all the power-impacted packets
                                        if packet in c:
                                            # check timing collision (whether packet is spared by preamble or not) 
                                            clear_pream_time_heard_here, not_enough_preamble = timingCollision(packet, nodes[other].packet, ocurring_now=False)

                                            # preamble is heard clearly only after the last ending interfering frame (not necessarily the last starting if sizes are different)
                                            clear_pream_time_heard = min(clear_pream_time_heard, clear_pream_time_heard_here)

                                            if not_enough_preamble:# packet missed
                                                locally_collided=True
                                                previous_frames_impacting_this_one.append((nodes[other].packet.nodeid,nodes[other].packet.ptype))
                                                if log_events:
                                                    MainLogger.info((self.nodeid,"col",packet.nodeid,packet.nodeid,env.now))
                                            else:
                                                receiving_packet = True
                                        else:
                                            receiving_packet = True

                                        # if preamble clear, start receiving it until a following packet comes impacting (further in loop)
                                        if receiving_packet:
                                            packet.receiving[self.nodeid]=True
                                            packet.receiving_start[self.nodeid]=env.now

                                        # impact on others (preceding)
                                        if nodes[other].packet in c:
                                            if log_events:
                                                MainLogger.info((self.nodeid,"col",packet.nodeid,nodes[other].packet.nodeid,env.now))
                                            previous_frames_impacted_by_this_one.append((nodes[other].packet.nodeid,nodes[other].packet.ptype))
                                            # unlock if impacted:
                                            nodes[other].packet.receiving[self.nodeid]=False

                                    else: # both collide
                                        if log_events:
                                            MainLogger.info((self.nodeid,"col",packet.nodeid,nodes[other].packet.nodeid,env.now))
                                            MainLogger.info((self.nodeid,"col",packet.nodeid,packet.nodeid,env.now))
                                        previous_frames_impacting_this_one.append((nodes[other].packet.nodeid,nodes[other].packet.ptype))
                                        previous_frames_impacted_by_this_one.append((nodes[other].packet.nodeid,nodes[other].packet.ptype))
                                        locally_collided=True

                        # store info for analysis at the end of the listen
                        self.heard_frames.append({
                            "id":len(self.heard_frames),
                            "nodeid":packet.nodeid,
                            "toa":packet.rectime,
                            "is_RTS":packet.ptype == rtsPacketType,
                            "dataPayloadSize_in_RTS":rng.integers(0,max_payload_size+1) if packet.dataPayloadSize==(profiles[self.profile]["CANL22_RTS_hdr_size"]+1) else packet.dataPayloadSize, # if Data of RTS size, a random size will be detected
                            "dataPayloadSize_in_EH":packet.dataPayloadSize, # in explicit header
                            "start":packet.addTime,
                            "interfering":previous_frames_impacted_by_this_one,
                            "interfered_by":previous_frames_impacting_this_one,
                            "clear_pream_time_heard":clear_pream_time_heard
                        })

                    else: # first poa, add it to heard_frames, will be discarded afterwards if preamble not heard enough
                        self.heard_frames.append({
                            "id":len(self.heard_frames),
                            "nodeid":packet.nodeid,
                            "toa":packet.rectime,
                            "is_RTS":packet.ptype == rtsPacketType,
                            "dataPayloadSize_in_RTS":rng.integers(0,max_payload_size+1) if packet.dataPayloadSize==(profiles[self.profile]["CANL22_RTS_hdr_size"]+1) else packet.dataPayloadSize, # if Data of RTS size, a random size will be detected
                            "dataPayloadSize_in_EH":packet.dataPayloadSize, # in explicit header
                            "start":packet.addTime,
                            "interfering":[],# what happenned prior to 0 is not on air anymore
                            "interfered_by":[],# what happenned prior to 0 is not on air anymore, could have been captured just a few syms of preamble and be heard now
                            "clear_pream_time_heard":clear_pream_time_heard                            
                        })

                    if not locally_collided:
                        if log_events:
                            MainLogger.info((self.nodeid,"rx",packet.nodeid,env.now))



    # function to generate future traffic
    # called when previous tx process finishes, but only gen frames if necessary
    def generate_new_frames(self):
        global nrScheduled

        # while last frame gen was in past  
        while env.now>self.next_gen_time or len(self.queue) == 0 :

            # if we enter too many times here, we need to drop
            if len(self.queue) == max_queue_size :
                self.n_dropped+=1
            else:
                self.queue.append(0)

            self.queue = [f-1 for f in self.queue]

            # produce next packet
            if experiment==6:
                #normally 9 nodes with 100ms delay between each node
                inter_gen_delay=self.cycle*self.period-env.now+self.nodeid*100
            elif experiment==7:
                #normally 5 nodes with 500ms delay between each node
                inter_gen_delay=self.cycle*self.period-env.now+self.nodeid*500            
            else:
                if self.distrib==perioDistribType:
                    inter_gen_delay=self.period
                if self.distrib==expoDistribType:
                    inter_gen_delay = rng.exponential(float(self.period))
                    # transmit_wait = rng.expovariate(1.0/float(self.period))
                if self.distrib==uniformDistribType:
                    inter_gen_delay = rng.uniform(max(2000,self.period-5000),self.period+5000)        
            
            # next_gen_time=last_gen_time+inter_gen_delay
            if shuffle_start and self.next_gen_time==-1:
                self.next_gen_time+=rng.uniform(0,self.period)
            self.next_gen_time+=inter_gen_delay
            self.gen_times.append(self.next_gen_time)

            if self.cluster!=-2: # if it is not an interferer, but a legit node
                nrScheduled += 1


        ################ set next packet
        oldest_gen_non_dropped = self.queue.pop(0)
        self.packet.setDataPayloadSize()
        self.packet.setPacketType(dataPacketType)                
        self.cycle = self.cycle + 1
        self.n_payload_gen += self.packet.dataPayloadSize 


        if env.now < self.gen_times[oldest_gen_non_dropped]: # oldest gen is in future, lets wait

            transmit_wait=self.gen_times[oldest_gen_non_dropped] - env.now
            yield env.timeout(transmit_wait)
            self.queued_time = env.now
            # self.want_transmit_time = env.now

        else: # oldest gen is in past, has been queued
            self.queued_time = self.gen_times[oldest_gen_non_dropped]

        # change state and go attemping a tx
        self.MAC_state=want_transmit


#
# this class creates a packet (associated with a node)
# it also sets all parameters
#
class myPacket():
    ## node ID, default packet length, distances to the GWs
    def __init__(self, nodeid, plen, distance_GWs):

        self.nodeid = nodeid

        self.dist_GWs=distance_GWs

        # (obsolete) randomize configuration values
        if lora24GHz:
            self.sf = rng.integers(5,13)
            self.bw = rng.choice([203.125, 406.250, 812.5, 1625])
        else:    
            self.sf = rng.integers(6,13)
            self.bw = rng.choice([125, 250, 500])
        self.cr = rng.integers(1,5)

        # for certain experiments override these
        if experiment==1 or experiment == 0:
            self.sf = 12
            self.cr = 4
            if lora24GHz:
                self.bw = 203.125
            else:    
                self.bw = 125

        # for certain experiments override these
        if experiment==2:
            if lora24GHz:
                self.sf = 5
                self.cr = 1
                self.bw = 1625            
            else:
                self.sf = 6
                self.cr = 1
                self.bw = 500

        # CURRENT EXPERIMENT IS 4 (2025-04-10)
        if experiment in [4,6,7]:
            if lora24GHz:
                self.bw = 203.125            
            else:
                self.bw = 125
            self.sf = exp4SF # CURRENTLY 12
            self.cr = 1    

        if experiment == "SF7BW500CAD4":
            self.bw = 500
            self.sf = 7
            self.cr = 1    # CR is 1, 2, 3 or 4 for respective coding rates 4/5, 4/6, 4/7 or 4/8

            
        # default not applied: adjust tx power according to expected path-loss 
        # a la np-CECADA (Np-CECADA: Enhancing Ubiquitous Connectivity of LoRa Networks,” in MASS’2021)
        if transmit_at_adjusted_power: 

            pred_noise_dB = 0
            if gaussian_noise:
                pred_noise_dB = 2*constants.noise_mu_dB

            pred_rayleigh_dB = 0
            if rayleigh_fading:
                pred_rayleigh_dB = rayleigh_mean_dB


            pred_local_obstruction_dB = 0 
            if locally_obstructed:
                pred_local_obstruction_dB = 2*constants.obstruction_dB_GW

            pred_buildings_atten_dB = 0 
            if buildings_attenuated:
                pred_buildings_atten_dB = buildings_km*distance_GWs/1000*constants.loss_per_building_GW

            # matrix of every path loss
            for txp in constants.power_levels:
                rss_GWs=np.minimum(txp , txp + constants.GL_GW - constants.Lpld0 - 10*gamma_GW*np.log10(distance_GWs/constants.d0) - pred_rayleigh_dB - pred_local_obstruction_dB - pred_buildings_atten_dB)
                heard_at_gateways = (rss_GWs - pred_noise_dB > get_sensitivity(self.sf,self.bw))
                if heard_at_gateways.any(): # one gw would receive it, enough
                    break
                if txp == Ptx:
                    break

            self.txpow = txp

        else:
            self.txpow = Ptx


        # following portion is a little obsolete
        ########################
        ########################
        ########################

        # for experiment 3 find the best setting
        # OBS, some hardcoded values
        Prx = self.txpow    ## zero path loss by default

        # log-shadow at init
        Lpl = constants.Lpld0 + 10*gamma_GW*np.log10(distance_GWs/constants.d0)
        Prx = np.minimum(self.txpow, self.txpow + constants.GL_GW - Lpl)

        #TODO for lora24GHz
        if (experiment == 3) or (experiment == 5):
            minairtime = 9999
            minsf = 0
            minbw = 0

            # print("Prx:", Prx)

            for i in range(0,6):
                for j in range(1,4):
                    if (sensi[i,j] < Prx).any():
                        self.sf = int(sensi[i,0])
                        if j==1:
                            self.bw = 125
                        elif j==2:
                            self.bw = 250
                        else:
                            self.bw=500
                        at = airtime(self.sf, 1, plen, self.bw)
                        if at < minairtime:
                            minairtime = at
                            minsf = self.sf
                            minbw = self.bw
                            minsensi = sensi[i, j]
            if (minairtime == 9999):
                # print("does not reach base station")
                exit(-1)
            # print("best sf:", minsf, " best bw: ", minbw, "best airtime:", minairtime)
            self.rectime = minairtime
            self.sf = minsf
            self.bw = minbw
            self.cr = 1

            if experiment == 5:
                # reduce the txpower if there's room left
                self.txpow = max(2, self.txpow - math.floor(np.min(Prx) - minsensi))
                Prx = self.txpow - GL - Lpl
                # print( 'minsesi {} best txpow {}'.format(minsensi, self.txpow))

        
        # end of obsolete portion
        ########################
        ########################
        ########################

        self.symTime = (2.0**self.sf)/self.bw

        # init values
        self.pl = plen
        self.rss_GWs = Prx # *np.ones(distance_GWs.shape)
        
        # Path loss exponents to neighs
        self.gamma_array = np.zeros((distance_matrix[self.nodeid].shape))
        if normal_gamma_ED:
            self.gamma_array = rng.normal(gamma_ED,sigma_gamma_ED,self.gamma_array.shape)
        else:
            self.gamma_array += gamma_ED


        #init interf and lock variable for rx in devs and gws
        self.interf_mw = np.zeros((distance_matrix[self.nodeid].shape))
        self.interf_mw_GWs = np.zeros((self.dist_GWs.shape))

        self.receiving = np.full((distance_matrix[self.nodeid].shape), False)
        self.receiving_GWs = np.full((self.dist_GWs.shape), False) 

        self.receiving_start = np.zeros((distance_matrix[self.nodeid].shape))
        self.receiving_start_GWs = np.zeros((self.dist_GWs.shape))


        #################### OLD USELESS ###############################################
        # # frequencies: lower bound + number of 61 Hz steps
        # if lora24GHz:
        #     self.freq = 2403000000 + rng.integers(0,2622951)
        # else:
        #     self.freq = 860000000 + rng.integers(0,2622951)

        # # for certain experiments override these and
        # # choose some random frequences
        # if experiment == 1:
        #     if lora24GHz:
        #         self.freq = rng.choice([2403000000, 2425000000, 2479000000])
        #     else:
        #         self.freq = rng.choice([860000000, 864000000, 868000000])
        # else:
        #     if lora24GHz:
        #         self.freq = 2403000000
        #     else:
        #         self.freq = 860000000
    

        self.ptype = dataPacketType
        
        #self.data_len keeps the payload length of a data packet
        #self.pl will be used in presence of RTS to discriminate the current packet length which can either be 
        #data_len or 5 (the size of an RTS packet)
        self.data_len=plen
        self.dataPayloadSize=self.data_len

        if lora24GHz:
            Npream = 12     
            if self.sf < 7:
                self.Tpream = (Npream + 6.25)*self.symTime
            else:
                self.Tpream = (Npream + 4.25)*self.symTime                 
        else:        
            Npream = 8     # number of preamble symbol (12.25     from Utz paper) 
            self.Tpream = (Npream + 4.25)*self.symTime        

        #init, will be updated by self.setDataPayloadSize() and setPacketType()
        self.rectime = airtime(self.sf,self.cr,self.pl,self.bw)

        # min # of preamble symbols that should be heard to enable detection
        self.Tmin_preamb_heard=self.symTime *  detection_min_preamb_symb

        # denote if packet is collided (init of vars)
        self.collided = 0
        self.collided_at_GWs = [0]*len(distance_GWs)
        self.processed = 0 # not in use in the current version

    #
    # give payload size according to distribution
    def setDataPayloadSize(self):
        if variablePayloadSize:
            if normalPayloadSize :
                self.dataPayloadSize=rng.normal(normaldist_mean_payload_size,normaldist_sigma_payload_size,1).astype('int').clip(dist_min_payload_size,dist_max_payload_size)[0]
            else: # uniform
                self.dataPayloadSize=rng.integers(dist_min_payload_size,dist_max_payload_size+1)

            if profiles[nodes[self.nodeid].profile]["protocol"]=="CANL22":
            # if CANL22: #depends on scenario, data length included in header or in data... 
                self.dataPayloadSize+=profiles[nodes[self.nodeid].profile]["CANL22_data_hdr_size"]

    #
    ## change packet type and size accordingly
    def setPacketType(self,ptype):
        self.ptype = ptype
        
        if ptype == rtsPacketType:
            self.pl=5
            # if CANL22:
            if profiles[nodes[self.nodeid].profile]["protocol"]=="CANL22":
                self.pl=profiles[nodes[self.nodeid].profile]["CANL22_RTS_hdr_size"]+1
                self.rectime = airtime(self.sf,self.cr,self.pl,self.bw,explicit=(profiles[nodes[self.nodeid].profile]["CANL22_RTS_PHY_HDR"]==0))         
            else:
                self.rectime = airtime(self.sf,self.cr,self.pl,self.bw,explicit=(LoRa_PHY_HDR==0))         
        else:
            self.pl=self.dataPayloadSize # self.data_len
            self.rectime = airtime(self.sf,self.cr,self.pl,self.bw,explicit=(LoRa_PHY_HDR==0))


    # set/update a random rx array to all neighbors
    def repropagate(self, first_time=False):  

        ## fast local noise
        self.noise_dB = np.zeros((self.dist_GWs.shape)) # to GWs
        self.noise_dB_arr = np.zeros((distance_matrix[self.nodeid].shape)) # to all neighs
        if gaussian_noise:
            self.noise_dB = np.clip(rng.normal(constants.noise_mu_dB,constants.noise_sigma_dB,self.dist_GWs.shape),0,2*constants.noise_mu_dB)
            self.noise_dB_arr = np.clip(rng.normal(constants.noise_mu_dB,constants.noise_sigma_dB,self.noise_dB_arr.shape),0,2*constants.noise_mu_dB)

        ## fast rayleigh multipath fading
        rayleigh_dB = np.zeros((self.dist_GWs.shape)) # to GWs
        rayleigh_dB_arr = np.zeros((distance_matrix[self.nodeid].shape))
        
        if rayleigh_fading: # gain is exponential (of mean 1, i.e. 0 dB)
            # rayleigh_dB = rng.rayleigh(scale=np.sqrt(2 / np.pi)*rayleigh_mean_dB,size=self.dist_GWs.shape) - rayleigh_mean_dB
            # rayleigh_dB_arr = rng.rayleigh(scale=np.sqrt(2 / np.pi)*rayleigh_mean_dB,size=rayleigh_dB_arr.shape) - rayleigh_mean_dB
            rayleigh_dB = -10*np.log10(rng.exponential(scale=10**(.1*rayleigh_mean_dB), size=self.dist_GWs.shape))
            rayleigh_dB_arr = -10*np.log10(rng.exponential(scale=10**(.1*rayleigh_mean_dB), size=rayleigh_dB_arr.shape))

        if first_time: # prepare coherent things only the first propagate

            # local shadowing
            self.local_obstruction_dB = np.zeros((self.dist_GWs.shape)) # to GWs
            self.local_obstruction_dB_arr = np.zeros((distance_matrix[self.nodeid].shape))

            if locally_obstructed:
                self.local_obstruction_dB = np.clip(rng.normal(constants.obstruction_dB_GW,constants.obstruction_dB_GW,self.dist_GWs.shape),0,2*constants.obstruction_dB_GW)
                self.local_obstruction_dB_arr = np.clip(rng.normal(constants.obstruction_dB,constants.obstruction_dB,self.local_obstruction_dB_arr.shape),0,2*constants.obstruction_dB)

            # link shadowing accounting for buimdings
            self.buildings_atten_dB = np.zeros((self.dist_GWs.shape)) # to GWs
            self.buildings_atten_dB_arr = np.zeros((distance_matrix[self.nodeid].shape))

            if buildings_attenuated:
                self.buildings_atten_dB = buildings_km*self.dist_GWs/1000*constants.loss_per_building_GW
                self.buildings_atten_dB_arr = buildings_km*distance_matrix[self.nodeid]/1000*constants.loss_per_building      

        # matrix of every received signal strength accounting for path loss, coherent shadowing and fast mp fading 
        self.rx_array=np.clip(-1000,self.txpow,self.txpow + constants.GL - constants.Lpld0 - 10*self.gamma_array*np.log10(distance_matrix[self.nodeid]/constants.d0) - rayleigh_dB_arr - self.local_obstruction_dB_arr - self.buildings_atten_dB_arr)
        # - self.noise_dB_arr 
        self.rss_GWs=np.minimum(self.txpow,self.txpow + constants.GL_GW - constants.Lpld0 - 10*gamma_GW*np.log10(self.dist_GWs/constants.d0) - rayleigh_dB - self.local_obstruction_dB - self.buildings_atten_dB)
        #- self.noise_dB 

        # reset interference caused by packet, and who is locked on it
        self.interf_mw = np.zeros((distance_matrix[self.nodeid].shape)) # interf suff by this packet on each listening dev
        self.interf_mw_GWs = np.zeros((self.dist_GWs.shape)) # # interf suff by this packet on each gw

        self.receiving = np.full((distance_matrix[self.nodeid].shape), False)
        self.receiving_GWs = np.full((self.dist_GWs.shape), False) 


    # remove my interference when I leave
    def stop_interfering(self):
        # if somebody else is transmitting
        if len(packetsOnAir)>0:
            for other in packetsOnAir:
                if frequencyCollision(self, nodes[other].packet): # hyp: channels are orthogonal
                    nodes[other].packet.interf_mw_GWs -= 10**(.1*self.rss_GWs)

        for node in nodes:
            # if CANL22:
            if profiles[node.profile]["protocol"]=="CANL22":
                if node.MAC_state in [CANL_listen1, CANL_listen2]:
                    if check_heard(self,node.nodeid):
                        this_mw_this_dev = 10**(.1*(self.rx_array[node.nodeid]))

                        for other in packetsOnAir:
                            if check_heard(nodes[other].packet,node.nodeid):

                                other_signal_mw = 10**(.1*(nodes[other].packet.rx_array[node.nodeid]))

                                # self.interf_mw[node.nodeid] -= other_signal_mw ===> will be reset by repropagate

                                nodes[other].packet.interf_mw[node.nodeid] -= this_mw_this_dev


# determines belonging to a same cluster
def same_cluster(devid,other_devid):
    node = nodes[devid]
    other = nodes[other_devid]

    if node.cluster not in [-1,-2]:# cluster -1 is the no-cluster, -2 is the cluster of externals
        if node.cluster==other.cluster:
            return True
    return False


# compute the CAD prob of success (true positive) 
def get_CAD_prob(distance,txpower,in_same_cluster=False):
    if in_same_cluster:
        return(CAD_diplos[txpower][1][round(distance)-1])
    return(CAD_diplos[txpower][0][round(distance)-1])

# pre-simulate the CAD proba of success as the smoothed reception rate of 100 packets at each meter and interpolated
# depends on propa variables and lora params
def prepare_diplo(loss_per_building, aper, sf, bw):
    diplos = {}
    for txp in constants.power_levels:
        my_matshape=(max(2000,int(2.2*maxDist)),100)

        distances_OL=np.linspace(1,my_matshape[0],my_matshape[0])
        distances=np.vstack([distances_OL]*my_matshape[1]).T

        noise=0
        if gaussian_noise:
            noise = np.clip(rng.normal(constants.noise_mu_dB,constants.noise_sigma_dB,my_matshape),0,2*constants.noise_mu_dB)

        rayleigh=0
        if rayleigh_fading:
            # rayleigh = rng.rayleigh(scale=np.sqrt(2 / np.pi)*rayleigh_mean_dB,size=my_matshape) - rayleigh_mean_dB
            rayleigh = -10*np.log10(rng.exponential(scale=10**(.1*rayleigh_mean_dB), size=my_matshape))


        apertures=np.clip(rng.normal(aper,aper,my_matshape),0,2*aper)
        buildings_atten=buildings_km*distances/1000*loss_per_building + apertures

        # Path loss exponents EDED for CADs, regardless of clustering
        CAD_gamma_EDs = np.zeros(my_matshape)
        if CAD_normal_gamma_ED:
            CAD_gamma_EDs = rng.normal(gamma_ED+CAD_gamma_ED_delta,CAD_sigma_gamma_ED,CAD_gamma_EDs.shape)
        else:
            CAD_gamma_EDs += gamma_ED+CAD_gamma_ED_delta


        diplos[txp] = {}
        for intra_cluster_scenario in [0,1]:


            Lpl = constants.Lpld0 + 10*(CAD_gamma_EDs-intra_cluster_gamma_gain*intra_cluster_scenario)*np.log10(distances/constants.d0)

            CAD_Prx = np.clip(-1000,txp,txp + constants.GL - Lpl - noise - rayleigh - buildings_atten)

            # 
            predicted_PDRs = np.count_nonzero((CAD_Prx>get_sensitivity(sf,bw,receiver_type="DEVICE")),axis=1)

            # >>> scipy.version.version : '1.6.2' requires window_length to be odd, not version 1.12.0 anymore
            window_length = max(50,int(2.2*maxDist/20))
            if window_length%2==0:
                window_length+=1

            y = savgol_filter(predicted_PDRs, window_length, 2)

            if max(y)<100:
                y=y/max(y)*100.1
            
            val_id=0
            while y[val_id]<100:
                y[val_id] = 101
                val_id+=1

            y = np.clip(y,a_min=0, a_max=100)

            diplos[txp][intra_cluster_scenario] = y

    return diplos



#
## CAD mechanism "requires" energy is received from a transmitter during all the CAD duration, hence we need a copy of the global on-air list at beginning of CAD
def start_CAD(nodeid):
    return [ transmitter for transmitter in packetsOnAir if nodes[transmitter].ch == nodes[nodeid].ch]

#
## compute CAD success for transmissions assumed continuous during full period  
def stop_CAD(nodeid,on_air_at_CAD_start):
    on_air_at_CAD_stop = [ transmitter for transmitter in packetsOnAir if nodes[transmitter].ch == nodes[nodeid].ch]

    #Hyp: no blank of less than CAD symbols between two tx of same device (if device n is transmitting at start and at stops => it is assumed to have been transmitting during all the CAD time)
    for devid in on_air_at_CAD_stop:
        if devid in on_air_at_CAD_start:
            if var_CAD_prob: #general case 
                if full_distances:
                    if rng.random()*100 <= get_CAD_prob(distance_matrix[nodeid][devid],nodes[devid].packet.txpow,same_cluster(nodeid,devid)):
                        if log_events:
                            MainLogger.info((nodeid,"CAD+",env.now))
                        return (True)
                else:
                    if rng.random()*100 <= get_CAD_prob(nodes[nodeid].dist_to_center,nodes[devid].packet.txpow,same_cluster(nodeid,devid)):
                        if log_events:
                            MainLogger.info((nodeid,"CAD+",env.now))                        
                        return (True)                    
            else: # for testing e.g. all CAD have 80% success rate
                if rng.random()*100 <= CAD_prob and CAD_prob!=0:
                    if log_events:
                        MainLogger.info((nodeid,"CAD+",env.now))                    
                    return (True)                    
    if log_events:
        MainLogger.info((nodeid,"CAD-",env.now))
    return False              

#
## Build a numpy array of distances inter devices
def build_dist_mat(topo):
    dist_mat=np.zeros((nrNodes,nrNodes))

    for i in range(nrNodes):
        for j in range(nrNodes):
            if j!=i:
                dist_mat[i][j]=(
                    (topo['nodes'][i]['x']-topo['nodes'][j]['x'])**2+
                    (topo['nodes'][i]['y']-topo['nodes'][j]['y'])**2
                    )**(1/2)
            else:
                dist_mat[i][j]=0.01 #avoid log10(0) (aka I am 1cm away from myself)

    return dist_mat



#
# main discrete event loop, runs for each node
# a global list of packet being processed while new generations occur
#
def communicate(env,node):

    ###### No need to redeclare global lists, lists are mutable
    # global packetsOnAir
    # global nodes
    # global channel_busy_rts
    # global channel_busy_data
    
    global endSim

    #### variables updated with the transmissions
    global nrLost
    global nrCollisions
    global nrCaptures
    global nrReceived
    global nrProcessed
    global nrSent
    global nrScheduled

    global nrRTSLost
    global nrRTSCollisions
    global nrRTSReceived
    global nrRTSProcessed
                    
    global n_transmit
    global inter_transmit_time
    global last_transmit_time
    global lastDisplayTime  

    # global ideal_latest_start,ideal_latest_time

    last_transmit_time=0


    #Main loop for each node
    while True:
        
        ###////////////////////////////////////////////////////////
        # CANL - xCANL_CAD - variants with RTS                    /
        ###////////////////////////////////////////////////////////        
        if profiles[node.profile]["protocol"]=="CANL22":
        # if CANL22 and (node.cluster!=-2):
            ###########################################################
            # schedule_tx -> want_transmit                            #
            # want_transmit -> listen_1                               #
            # listen_1 -> NAV_state | send_RTS                        #
            # NAV_state -> want_transmit                              #
            # send_RTS -> listen_2 | send_DATA                        #
            # listen_2 -> NAV_state | send_DATA                       #
            # send_DATA -> schedule_tx                                #
            ###########################################################

            ###########################################################
            # schedule_tx -> want_transmit                            #
            ###########################################################
            if node.MAC_state==schedule_tx:
            
                ## scheduling a new generation
                yield env.process(node.generate_new_frames())


            ###########################################################
            # want_transmit -> listen_1                               #
            ###########################################################
            if node.MAC_state==want_transmit:# and node.packet.ptype==dataPacketType:
                if node.n_retry==0: # no more trials possible, abort or ALOHA
                    node.remainder = -1
                    if profiles[node.profile]["abort_after_retries"]:
                        node.n_aborted = node.n_aborted +1
                        #reset for sending a new packet              
                        
                        node.total_retry += profiles[node.profile]["n_retry"]
                        node.n_retry=profiles[node.profile]["n_retry"]
                        node.nav=0

                        node.update_available_channels()

                        node.MAC_state=schedule_tx
                        # node.Wbusy_BE=Wbusy_BE
                    else:
                        yield env.timeout(node.nav_period)            
                        node.backoff=rng.integers(profiles[node.profile]["Wcanl_min_sym"],profiles[node.profile]["Wcanl_max_sym"]+1)*node.packet.symTime# 64 == 2**Wbusy_maxBE
                        yield env.timeout( node.backoff)
                        node.kamikaze += 1
                        node.MAC_state=CANL_send_DATA
                        node.backoff=0
                else:                    
                    if node.nav!=0:
                        #reset nav to start again a complete CA procedure
                        node.nav=0
                        node.nav_period = node.nav_period_default
                    else:
                        #this is an initial transmit attempt
                        node.want_transmit_time=env.now
                        n_transmit = n_transmit + 1
                        if n_transmit > 1:
                            current_inter_transmit_time = env.now - last_transmit_time
                            inter_transmit_time += current_inter_transmit_time
                        last_transmit_time = env.now        
                        
                    
                    if profiles[node.profile]["CANL22_P"]!=0:
                        #determine if the node transmits data right after RTS or after a listen 2 phase
                        node.my_P=rng.integers(0,101)

                    # CAD before LISTEN, optional, default 2022 not applied
                    channel_found_busy=False                    
                    CAD_time=0
                    if profiles[node.profile]["CANL22_check_busy"] and node.channel_just_changed:
                        node.channel_just_changed = False
                        node.MAC_state=CANL_CAD
                        # node.n_CAD = node.n_CAD + 1

                        if profiles[node.profile]["nCAD_DIFS"]>0:
                            no_difs_has_been_found_clear_yet = True
                            while no_difs_has_been_found_clear_yet:


                                for CAD_id in range(profiles[node.profile]["nCAD_DIFS"]):
                                    node.n_CAD = node.n_CAD + 1

                                    if env.now - node.radio_on_last > node.radio_wake_up_time:
                                        yield env.timeout(node.radio_wake_up_time)
                                        node.radio_wake_up_number +=1

                                    ### Wait CAD Duration
                                    CAD_time=node.packet.symTime*nCadSym
                                    CAD_start_time = env.now

                                    on_air_at_CAD_start=start_CAD(node.nodeid) # simulator stores transmitters at CAD start
                                    yield env.timeout(CAD_time)
                                    channel_found_busy=stop_CAD(node.nodeid,on_air_at_CAD_start) # determines CAD + or -
                                    yield env.timeout(node.packet.symTime/2) # process CAD
                                    listen_log.append((env.now,CAD_start_time,CAD_time,node.nodeid,node.cycle,int(node.ch))) ## convert numpy.int64 to int for jsonify
                                    node.radio_on_last = env.now

                                    if channel_found_busy or (CAD_id==(profiles[node.profile]["nCAD_DIFS"]-1)) :
                                        break
                                    

                                    to_wait_bw_cads = node.packet.symTime*profiles[node.profile]["DIFS_inter_CAD_sym"]

                                    if to_wait_bw_cads > node.radio_wake_up_time:
                                        node.radio_on_last = -10000 # force re-wake_up of radio
                                        to_wait_bw_cads -= node.radio_wake_up_time

                                    yield env.timeout(to_wait_bw_cads)


                                if channel_found_busy: # DIFS went busy
                                    if profiles[node.profile]["CH_DIFS"]: #TODO: new proto xCH_CAD+BO
                                        if node.update_channel():
                                            yield env.timeout(node.radio_config_time)
                                            node.radio_config_number += 1
                                        else: # no change possible, break anyway, but before, reset avail chan
                                            
                                            # TODO: is it really better to maintain difs discarded chans?
                                            # currently: only reset if doing NAV/CH & retry

                                            # if profiles[node.profile]["CH_DIFS"]
                                            # node.update_available_channels() 

                                            break # while
                                    else: # Do single difs, break anyway
                                        break # while                        
                                else: # chan found free
                                    break # while    
        
                    if channel_found_busy:
                        node.nav+=1         
                        #will go into NAV
                        node.MAC_state=CANL_NAV_state  
                        
                    else: # go channel clear state, then

                        #change packet type to get the correct time-on-air for calculation
                        node.packet.setPacketType(rtsPacketType)


                        if profiles[node.profile]["BO_when_clear"]:
                            
                            if profiles[node.profile]["active_clear_BO"]: #perform CADs during BO
                                if node.remainder == -1:

                                    # node.backoff=rng.integers(Wclear_min_sym,2**node.Wclear_BE+1)                        
                                    node.remainder=rng.integers(profiles[node.profile]["Wclear_min_CAD"],profiles[node.profile]["Wclear_max_CAD"]+1)                        

                                for CAD_id in range(node.remainder):
                                    node.n_CAD = node.n_CAD + 1

                                    if env.now - node.radio_on_last > node.radio_wake_up_time:
                                        yield env.timeout(node.radio_wake_up_time)
                                        node.radio_wake_up_number +=1

                                    ### Wait CAD Duration
                                    CAD_time=node.packet.symTime*nCadSym
                                    CAD_start_time = env.now


                                    on_air_at_CAD_start=start_CAD(node.nodeid) # simulator stores transmitters at CAD start
                                    yield env.timeout(CAD_time)
                                    channel_found_busy=stop_CAD(node.nodeid,on_air_at_CAD_start) # determines CAD + or -
                                    yield env.timeout(node.packet.symTime/2) # process CAD
                                    listen_log.append((env.now,CAD_start_time,CAD_time,node.nodeid,node.cycle,int(node.ch)))
                                    node.radio_on_last = env.now                                

                                    if channel_found_busy:
                                        # node.n_retry = node.n_retry - 1
                                        node.I_heard_activity=True

                                        break

                                    if (CAD_id==(node.remainder-1)) : # all clear, go transmit
                                        break

                                    to_wait_bw_cads = node.packet.symTime*profiles[node.profile]["BO_inter_CAD_sym"]

                                    if to_wait_bw_cads > node.radio_wake_up_time:
                                        node.radio_on_last = -10000 # force re-wake_up of radio
                                        to_wait_bw_cads -= node.radio_wake_up_time

                                    yield env.timeout(to_wait_bw_cads)

                                if channel_found_busy:
                                    # node.remainder = node.remainder-1-CAD_id
                                    node.remainder = node.remainder-CAD_id # last was busy, we need remainder clears to exit BO

                                else: # BO is successfully finished
                                    node.remainder = -1
                                    # print("BO is successfully finished",file=sys.stderr)

                            else: #passive BO when clear

                                node.backoff=node.packet.symTime*rng.integers(profiles[node.profile]["Wclear_min_sym"],profiles[node.profile]["Wclear_max_sym"]+1)
                                yield env.timeout(node.backoff)
                                node.backoff=0


                                if profiles[node.profile]["nCAD_DIFS"]>0:
                                    for CAD_id in range(profiles[node.profile]["nCAD_DIFS"]):
                                        node.n_CAD = node.n_CAD + 1

                                        if env.now - node.radio_on_last > node.radio_wake_up_time:
                                            yield env.timeout(node.radio_wake_up_time)
                                            node.radio_wake_up_number +=1

                                        ### Wait CAD Duration
                                        CAD_time=node.packet.symTime*nCadSym
                                        CAD_start_time = env.now

                                        on_air_at_CAD_start=start_CAD(node.nodeid) # simulator stores transmitters at CAD start
                                        yield env.timeout(CAD_time)
                                        channel_found_busy=stop_CAD(node.nodeid,on_air_at_CAD_start) # determines CAD + or -
                                        yield env.timeout(node.packet.symTime/2) # process CAD
                                        listen_log.append((env.now,CAD_start_time,CAD_time,node.nodeid,node.cycle,int(node.ch)))
                                        node.radio_on_last = env.now                                

                                        if channel_found_busy:
                                            # node.n_retry = node.n_retry - 1
                                            node.I_heard_activity=True
                                            break

                                        if (CAD_id==(profiles[node.profile]["nCAD_DIFS"]-1)) :
                                            break
                                        
                                        to_wait_bw_cads = node.packet.symTime*profiles[node.profile]["DIFS_inter_CAD_sym"]

                                        if to_wait_bw_cads > node.radio_wake_up_time:
                                            node.radio_on_last = -10000 # force re-wake_up of radio
                                            to_wait_bw_cads -= node.radio_wake_up_time

                                        yield env.timeout(to_wait_bw_cads)

                                else:
                                    channel_found_busy=False


                        # listen is done:
                        ## 1: when no BO is applied
                        ## 2: when BO gives BUSY
                        ## TODO: what if truely passive BO? 
                        ## TODO: if first CAD busy, frame probably started long ago; worth a Listen?
                        if (profiles[node.profile]["BO_when_clear"] and channel_found_busy) or not profiles[node.profile]["BO_when_clear"]:

                            if env.now - node.radio_on_last > node.radio_wake_up_time:
                                yield env.timeout(node.radio_wake_up_time)
                                node.radio_wake_up_number +=1

                            node.MAC_state=CANL_listen1

                            node.MAC_listen_start_time=env.now

                            node.start_listening()

                            # switch bw softer fairness or basic fairness (reduction of listen win max wrt n_retry)
                            if profiles[node.profile]["CANL22_softer_fair"]:
                                if node.n_retry<profiles[node.profile]["n_retry"]:
                                    CANL_win_max=max(profiles[node.profile]["CANL22_L1_min"],profiles[node.profile]["CANL22_L1_MAX"]-profiles[node.profile]["CANL22_fair_factor"]*(profiles[node.profile]["n_retry"]-node.n_retry-1))
                                else:
                                    CANL_win_max=max(profiles[node.profile]["CANL22_L1_min"],profiles[node.profile]["CANL22_L1_MAX"])
                            else:
                                CANL_win_max=max(profiles[node.profile]["CANL22_L1_min"],profiles[node.profile]["CANL22_L1_MAX"]-profiles[node.profile]["CANL22_fair_factor"]*(profiles[node.profile]["n_retry"]-node.n_retry))

                            # compute listen window
                            listen_time=rng.integers(
                                profiles[node.profile]["CANL22_L1_min"], 
                                CANL_win_max+1
                                )*node.packet.symTime #+node.packet.rectime#+.131


                            node.MAC_listen_end_time=env.now+listen_time
                            if log_events:
                                MainLogger.info((node.nodeid,"lis1_start",env.now))

                            node.listen_process = env.process(node.listen(listen_time))
                            remaining_to_listen = yield node.listen_process

                            reenter=0
                            while remaining_to_listen!=-1:

                                node.listen_process = env.process(node.listen(remaining_to_listen))
                                remaining_to_listen = yield node.listen_process

                                reenter+=1

                            node.radio_on_last = env.now

                            listen_log.append((env.now,node.MAC_listen_start_time,listen_time,node.nodeid,node.cycle,int(node.ch)))

                            node.listened_time= env.now - node.MAC_listen_start_time


                            
                        if (profiles[node.profile]["BO_when_clear"] and not channel_found_busy):
                            #go transmit without listening (transit by CANL_send_RTS state)
                            node.MAC_state=CANL_listen1
                            node.listened_time=0


            ###########################################################
            # listen_1 -> NAV_state | send_RTS                        #
            ###########################################################
            #node was in CANL_listen1
            #### WE stopped listening earlier! 
            if node.MAC_state==CANL_listen1 and node.listened_time!=-1:
                if log_events:
                    MainLogger.info((node.nodeid,"lis1_stop",node.MAC_listen_start_time+node.listened_time))
                node.total_listen_time = node.total_listen_time + node.listened_time
                
                #did we receive a DATA with a ValidHeader?
                if node.I_know_it_is_Data==True:
                    #nav period is the time-on-air of the maximum data size which is returned in node.nav or max_payload_size
                    node.nav_period=airtime(node.packet.sf,node.packet.cr,max_payload_size,node.packet.bw,explicit=(LoRa_PHY_HDR==0))-node.wait_PHY_interrupt
                    if profiles[node.profile]["Interrupts_on_header_valid"]:# we were able to find out the data size from its header
                        node.nav_period= airtime(node.packet.sf,node.packet.cr,node.next_payload_byte,node.packet.bw,explicit=(LoRa_PHY_HDR==0)) - node.wait_PHY_interrupt #node.data_rectime

                    #will go into NAV
                    node.MAC_state=CANL_NAV_state        
                    node.nav+=1        

                elif node.I_know_it_is_RTS==True:#it did receive an RTS
                    #we process this event at the end of the listening period, normally the RTS has been received in the past
                    node.nav_period= airtime(node.packet.sf,node.packet.cr,node.next_payload_byte,node.packet.bw,explicit=(LoRa_PHY_HDR==0)) #node.data_rectime
                    if profiles[node.profile]["CANL22_P"]!=0:
                        node.nav_period+=profiles[node.profile]["CANL22_L2"]

                    #go into NAV
                    node.MAC_state=CANL_NAV_state            
                    node.nav+=1


                elif node.I_heard_preamble or node.I_heard_activity:
                    #nav period is the time-on-air of the maximum data size which is returned in node.nav
                    node.nav_period=node.nav_period_default
                    #will go into NAV
                    node.MAC_state=CANL_NAV_state                
                    node.nav+=1

                else:# end of listening -> now send RTS (if any to send wrt scenario)

                    node.MAC_state=CANL_send_RTS

                    if node.packet.dataPayloadSize>profiles[node.profile]["CANL22_RTS_min_payload_size"]:# min is set very high in scenarios without RTS
                        
                        if env.now - node.radio_on_last > node.radio_wake_up_time:
                            yield env.timeout(node.radio_wake_up_time)
                            node.radio_wake_up_number +=1


                        node.n_rts_sent = node.n_rts_sent + 1
                        if (node.nodeid in packetsOnAir):
                            print("ERROR: RTS packet already in",file=sys.stderr)
                        # else:
                        heard_at_gateways = (node.packet.rss_GWs - node.packet.noise_dB > get_sensitivity(node.packet.sf,node.packet.bw))
                        if not heard_at_gateways.any(): # no gw receive it, too far
                            node.packet.lost = True
                            if full_distances: # on air anyway, let's check impact on neighbors
                                checkcollision(node.packet,heard_at_gateways)
                                packetsOnAir.append(node.nodeid)
                                node.packet.addTime = env.now
                        else:
                            node.packet.lost = False
                            checkcollision(node.packet,heard_at_gateways)
                            packetsOnAir.append(node.nodeid)
                            node.packet.addTime = env.now
                        if log_events:
                            MainLogger.info((node.nodeid,"TX_start",env.now))


                        channel_busy_rts[node.nodeid]=True
                        channel_log.append((env.now,node.packet.rectime,node.nodeid,node.cycle,int(node.ch)))
                        # print(node.nodeid,packetsOnAir,file=sys.stderr)
                        yield env.timeout(node.packet.rectime)
                        channel_busy_rts[node.nodeid]=False
                        if log_events:
                            MainLogger.info((node.nodeid,"TX_stop",env.now))

                        node.radio_on_last = env.now

                        yield env.timeout(.0000001) # avoid further data being caught at exact end of a listen

                        if node.packet.lost:
                            node.packet.collided=0
                            nrRTSLost += 1
                        else: # check if received and not collided somewhere
                            node.packet.collided=1
                            for gw_id in range(len(heard_at_gateways)):
                                if heard_at_gateways[gw_id]==1 and node.packet.collided_at_GWs[gw_id]==0:
                                    node.packet.collided=0
                                    break

                        if node.packet.collided == 1:
                            nrRTSCollisions = nrRTSCollisions +1
                        if node.packet.collided == 0 and not node.packet.lost:
                            nrRTSReceived = nrRTSReceived + 1
                            # print("node {} {}: RTS packet has been correctly transmitted".format(node.nodeid, env.now))
                        if node.packet.processed == 1:
                            nrRTSProcessed = nrRTSProcessed + 1

                        # complete packet has been received by base station
                        # can remove it
                        if (node.nodeid in packetsOnAir):
                            packetsOnAir.remove(node.nodeid)

                        # reset the packet
                        node.packet.stop_interfering()
                        node.packet.collided = 0
                        node.packet.collided_at_GWs = [0]*len(heard_at_gateways)
                        node.packet.processed = 0
                        node.packet.lost = False       
                        node.packet.repropagate()     
                    else: # payload is too short to consider sending RTS
                        pass


                # reinit listen params
                node.I_heard_preamble=False
                node.I_heard_activity=False
                node.I_know_it_is_Data=False
                node.I_know_it_is_RTS=False
                node.next_payload_byte=0
                node.listened_time=-1

                node.heard_frames=[]
                node.add_rx_time_opportunities=1 # 


            ###########################################################
            # NAV_state -> want_transmit                              #
            ###########################################################
            if node.MAC_state==CANL_NAV_state:
                #we arrive at the end of the nav period
                #so we try again from the beginning of the CANL22 procedure
                node.MAC_state=want_transmit
                node.packet.setPacketType(dataPacketType)
                # decrement retry counter
                node.n_retry = node.n_retry - 1   

                if node.n_retry > 0:
                    if node.nav_period>profiles[node.profile]["CANL22_MAX_NAV_x_chan_threshold"]:
                        if node.update_channel():
                            #printnode.nodeid, "change chan", env.now)
                            node.channel_just_changed = True
                            yield env.timeout(node.radio_config_time)
                            node.radio_config_number +=1
                        else:
                            yield env.timeout(node.nav_period)
                            node.update_available_channels()                          
                    else:
                        yield env.timeout(node.nav_period) 
                        node.update_available_channels()                          


            ###########################################################
            # send_RTS -> listen_2 | send_DATA                        #
            ###########################################################
            if node.MAC_state==CANL_send_RTS:
                if profiles[node.profile]["CANL22_P"]==0:
                    #default: no listening phase 2
                    node.MAC_state=CANL_send_DATA
                else:      
                    #printnode.nodeid, "enter listen 2", env.now)


                    if env.now - node.radio_on_last > node.radio_wake_up_time:
                        yield env.timeout(node.radio_wake_up_time)
                        node.radio_wake_up_number +=1

                    #we have sent RTS, and now we go for another listening period
                    node.MAC_state=CANL_listen_2   

                    #store time at which listening period began
                    node.MAC_listen_start_time=env.now
                    if log_events:
                        MainLogger.info((node.nodeid,"lis2_start",env.now))

                    node.start_listening()

                    #listen period is CANL22_L2*DIFS, with DIFS=preamble duration
                    listen_duration=profiles[node.profile]["CANL22_L2"]*node.packet.Tpream
                    node.MAC_listen_end_time=env.now+listen_duration


                    node.listen_process = env.process(node.listen(listen_time))
                    remaining_to_listen = yield node.listen_process

                    reenter=0
                    while remaining_to_listen!=-1:
                        node.listen_process = env.process(node.listen(remaining_to_listen))
                        remaining_to_listen = yield node.listen_process
                        reenter+=1

                    listen_log.append((env.now,node.MAC_listen_start_time,listen_duration,node.nodeid,node.cycle,int(node.ch)))

                    node.radio_on_last = env.now

                    node.listened_time= env.now - node.MAC_listen_start_time



            ###########################################################
            # listen_2 -> NAV_state | send_DATA                       #
            ###########################################################
            #### WE stopped listening earlier! See node.listened_time==-1 ###
            if node.MAC_state==CANL_listen2 and node.listened_time!=-1:
                if log_events:
                    MainLogger.info((node.nodeid,"lis2_stop",node.MAC_listen_start_time+node.listened_time))
                node.total_listen_time = node.total_listen_time + node.listened_time

                #did we receive a DATA with a ValidHeader?
                if node.I_know_it_is_Data==True:
                    node.nav_period=airtime(node.packet.sf,node.packet.cr,max_payload_size,node.packet.bw,explicit=(LoRa_PHY_HDR==0))-node.wait_PHY_interrupt

                    if profiles[node.profile]["Interrupts_on_header_valid"]:
                        node.nav_period= airtime(node.packet.sf,node.packet.cr,node.next_payload_byte,node.packet.bw,explicit=(LoRa_PHY_HDR==0)) - node.wait_PHY_interrupt #node.data_rectime

                    #will go into NAV
                    node.MAC_state=CANL_NAV_state
                    node.nav+=1



                elif node.I_know_it_is_RTS==True:#it did receive an RTS
                    node.nav_period= airtime(node.packet.sf,node.packet.cr,node.next_payload_byte,node.packet.bw,explicit=(LoRa_PHY_HDR==0)) #node.data_rectime
                    if profiles[node.profile]["CANL22_P"]!=0:
                        node.nav_period+=profiles[node.profile]["CANL22_L2"]

                    node.MAC_state=CANL_NAV_state
                    node.nav+=1
                    


                elif node.I_heard_preamble or node.I_heard_activity:
                    #nav period is the time-on-air of the maximum data size which is returned in node.nav
                    node.nav_period= node.nav_period_default
                    #will go into NAV
                    node.MAC_state=CANL_NAV_state
                    node.nav+=1

                else:    
                    node.MAC_state=CANL_send_DATA

                # reinit listen params
                node.I_heard_preamble=False
                node.I_heard_activity=False
                node.I_know_it_is_Data=False
                node.I_know_it_is_RTS=False
                node.next_payload_byte=0
                node.listened_time=-1

                node.heard_frames=[]
                node.add_rx_time_opportunities=1 # 

            ###########################################################
            # send_DATA -> schedule_tx                                #
            ###########################################################
            if node.MAC_state==CANL_send_DATA:      
                #printnode.nodeid, "enter send data", env.now)

                #change packet type to get the correct time-on-air
                node.packet.setPacketType(dataPacketType)

                if env.now - node.radio_on_last > node.radio_wake_up_time:
                    yield env.timeout(node.radio_wake_up_time)
                    node.radio_wake_up_number +=1


                # DATA time sending and receiving
                # DATA packet arrives -> add to base station
                node.n_data_sent = node.n_data_sent + 1
                node.n_payload_sent += node.packet.dataPayloadSize - profiles[node.profile]["CANL22_data_hdr_size"]
                nrSent+=1
                node.total_retry += profiles[node.profile]["n_retry"] - node.n_retry
                # node.latency = node.latency + (env.now-node.want_transmit_time)
                node.latency = node.latency + (env.now-node.queued_time)
                if (node.nodeid in packetsOnAir):
                    print("ERROR: DATA packet already in",file=sys.stderr)
                # else:

                heard_at_gateways = (node.packet.rss_GWs - node.packet.noise_dB > get_sensitivity(node.packet.sf,node.packet.bw))
                if not heard_at_gateways.any(): # no gw receive it, too far
                # if node.packet.rssi < get_sensitivity(node.packet.sf,node.packet.bw):
                    node.packet.lost = True
                    if full_distances:#on air anyway
                        checkcollision(node.packet,heard_at_gateways)
                        packetsOnAir.append(node.nodeid)
                        node.packet.addTime = env.now
                else:
                    node.packet.lost = False
                    checkcollision(node.packet,heard_at_gateways)
                    packetsOnAir.append(node.nodeid)
                    node.packet.addTime = env.now
                if log_events:
                    MainLogger.info((node.nodeid,"TX_start",env.now))


                channel_busy_data[node.nodeid]=True
                channel_log.append((env.now,node.packet.rectime,node.nodeid,node.cycle,int(node.ch)))
                # print(node.nodeid,packetsOnAir,file=sys.stderr)
                yield env.timeout(node.packet.rectime)
                channel_busy_data[node.nodeid]=False
                if log_events:
                    MainLogger.info((node.nodeid,"TX_stop",env.now))

                node.radio_on_last = env.now
                    
                if node.packet.lost:
                    node.packet.collided=0
                    nrLost += 1
                    node.n_lost+=1
                    # print("node {} {}: DATA packet was lost".format(node.nodeid, env.now))
                else: # check if received and not collided somewhere
                    node.packet.collided=1
                    for gw_id in range(len(heard_at_gateways)):
                        if heard_at_gateways[gw_id]==1 and node.packet.collided_at_GWs[gw_id]==0:
                            node.packet.collided=0
                            break

                if node.packet.collided == 1:
                    nrCollisions = nrCollisions + 1
                    node.n_collided+=1
                    # print("node {} {}: DATA packet was collided".format(node.nodeid, env.now))
                if node.packet.collided == 0 and not node.packet.lost:
                    nrReceived = nrReceived + 1
                    # print("node {} {}: DATA packet has been correctly transmitted".format(node.nodeid, env.now))
                    # current_latency=env.now-node.want_transmit_time
                    current_latency=env.now-node.queued_time
                    node.min_success_latency = min(node.min_success_latency, current_latency)
                    node.success_latency = node.success_latency + current_latency
                    node.n_data_success+=1
                    node.n_payload_success += node.packet.dataPayloadSize - profiles[node.profile]["CANL22_data_hdr_size"]
                if node.packet.processed == 1:
                    nrProcessed = nrProcessed + 1

                # complete packet has been received by base station
                # can remove it
                if (node.nodeid in packetsOnAir):
                    packetsOnAir.remove(node.nodeid)
                # reset the packet
                node.packet.stop_interfering()
                node.packet.collided = 0
                node.packet.collided_at_GWs = [0]*len(heard_at_gateways)
                node.packet.processed = 0
                node.packet.lost = False
                node.packet.repropagate()
                node.update_available_channels()
                node.last_tx_ch = node.ch
                if node.update_channel():
                    node.channel_just_changed = True
                    yield env.timeout(node.radio_config_time)
                    node.radio_config_number +=1

                node.n_retry=profiles[node.profile]["n_retry"]
                node.nav_period = node.nav_period_default
                node.nav=0
                node.MAC_state=schedule_tx





        ###////////////////////////////////////////////////////////
        # Ideal ideal_FIFO                                        /
        ###////////////////////////////////////////////////////////      
        elif profiles[node.profile]["protocol"] == "ideal_FIFO":

            ###########################################################
            # schedule_tx -> want_transmit                            #
            # want_transmit -> send_DATA                              #
            # send_DATA -> schedule_tx                                #
            ###########################################################

            ###########################################################
            # schedule_tx -> want_transmit                            #
            ###########################################################
            if node.MAC_state==schedule_tx:
                yield env.process(node.generate_new_frames())

            ###########################################################
            # want_transmit -> listen_1                               #
            ###########################################################
            if node.MAC_state==want_transmit:# and node.packet.ptype==dataPacketType:

                #this is an initial transmit attempt
                node.want_transmit_time=env.now
                n_transmit = n_transmit + 1
                if n_transmit > 1:
                    current_inter_transmit_time = env.now - last_transmit_time
                    inter_transmit_time += current_inter_transmit_time
                last_transmit_time = env.now 

                my_transmission_date=ideal_latest_start[node.ch]+ideal_latest_time[node.ch]+1/100000 #add 10 ns to avoid floating point error (4104.19200000001) and collisions!
                node.MAC_state=Ideal_send_DATA    
                ideal_latest_time[node.ch]=airtime(node.packet.sf,node.packet.cr,node.packet.dataPayloadSize,node.packet.bw,explicit=(LoRa_PHY_HDR==0))
                if node.want_transmit_time<my_transmission_date:
                    ideal_latest_start[node.ch]=my_transmission_date
                    #let's wait for the end of the previous tx:
                    yield env.timeout(my_transmission_date-node.want_transmit_time)
                else:
                    # ideal_latest_start[node.ch]=node.want_transmit_time
                    if env.now - node.radio_on_last > node.radio_wake_up_time:
                        yield env.timeout(node.radio_wake_up_time)
                        node.radio_wake_up_number +=1
                    ideal_latest_start[node.ch]=env.now
                    

            ###########################################################
            # send_DATA -> schedule_tx                                #
            ###########################################################
            if node.MAC_state==Ideal_send_DATA:        


                # DATA time sending and receiving
                # DATA packet arrives -> add to base station
                node.n_data_sent = node.n_data_sent + 1
                node.n_payload_sent += node.packet.dataPayloadSize
                nrSent+=1
                # node.latency = node.latency + (env.now-node.want_transmit_time)
                node.latency = node.latency + (env.now-node.queued_time)
                if (node.nodeid in packetsOnAir):
                    print("ERROR: DATA packet already in",file=sys.stderr)
                # else:

                heard_at_gateways = (node.packet.rss_GWs - node.packet.noise_dB > get_sensitivity(node.packet.sf,node.packet.bw))
                if not heard_at_gateways.any(): # no gw receive it, too far

                # if node.packet.rssi < get_sensitivity(node.packet.sf,node.packet.bw):
                    # print("node {}: DATA packet will be lost".format(node.nodeid))
                    node.packet.lost = True
                    if full_distances:#on air anyway
                        checkcollision(node.packet,heard_at_gateways)
                        packetsOnAir.append(node.nodeid)
                        node.packet.addTime = env.now
                else:
                    node.packet.lost = False
                    checkcollision(node.packet,heard_at_gateways)
                    packetsOnAir.append(node.nodeid)
                    node.packet.addTime = env.now
                if log_events:
                    MainLogger.info((node.nodeid,"TX_start",env.now))


                channel_busy_data[node.nodeid]=True
                channel_log.append((env.now,node.packet.rectime,node.nodeid,node.cycle,int(node.ch)))
                # print(node.nodeid,packetsOnAir,file=sys.stderr)
                yield env.timeout(node.packet.rectime)
                channel_busy_data[node.nodeid]=False
                if log_events:
                    MainLogger.info((node.nodeid,"TX_stop",env.now))

                node.radio_on_last = env.now
                    
                if node.packet.lost:
                    node.packet.collided=0
                    nrLost += 1
                    node.n_lost+=1
                    # print("node {} {}: DATA packet was lost".format(node.nodeid, env.now))
                else: # check if received and not collided somewhere
                    node.packet.collided=1
                    for gw_id in range(len(heard_at_gateways)):
                        if heard_at_gateways[gw_id]==1 and node.packet.collided_at_GWs[gw_id]==0:
                            node.packet.collided=0
                            break
                if node.packet.collided == 1:
                    nrCollisions = nrCollisions + 1
                    node.n_collided+=1
                    # print("node {} {}: DATA packet was collided".format(node.nodeid, env.now))
                if node.packet.collided == 0 and not node.packet.lost:
                    nrReceived = nrReceived + 1
                    # print("node {} {}: DATA packet has been correctly transmitted".format(node.nodeid, env.now))
                    # current_latency=env.now-node.want_transmit_time
                    current_latency=env.now-node.queued_time
                    node.min_success_latency = min(node.min_success_latency, current_latency)
                    node.success_latency = node.success_latency + current_latency
                    node.n_data_success+=1
                    node.n_payload_success += node.packet.dataPayloadSize
                if node.packet.processed == 1:
                    nrProcessed = nrProcessed + 1

                # complete packet has been received if not lost by base station
                # can remove it
                if (node.nodeid in packetsOnAir):
                    packetsOnAir.remove(node.nodeid)
                # reset the packet
                node.packet.stop_interfering()
                node.packet.collided = 0
                node.packet.collided_at_GWs = [0]*len(heard_at_gateways)
                node.packet.processed = 0
                node.packet.lost = False
                node.packet.repropagate()
                node.update_available_channels()
                node.last_tx_ch = node.ch                
                if node.update_channel():
                    yield env.timeout(node.radio_config_time)
                    node.radio_config_number +=1
                # node.n_retry=profiles[node.profile]["n_retry"] if "n_retry" in profiles[node.profile] else -1
                # node.nav=0
                node.MAC_state=schedule_tx


        ###////////////////////////////////////////////////////////                
        # CSMA APPROACHES                                         /
        # ALOHA protocol                                          /
        ###//////////////////////////////////////////////////////// 
        elif profiles[node.profile]["protocol"] in ["CAD+Backoff", "ALOHA"]:
            # Schedule next tx
            #########################################################################
            yield env.process(node.generate_new_frames())

            #########################################################################

            node.want_transmit_time=env.now
            
            if node.cluster!=-2:
                n_transmit = n_transmit + 1
                if n_transmit > 1:
                    current_inter_transmit_time = env.now - last_transmit_time
                    inter_transmit_time += current_inter_transmit_time
            last_transmit_time = env.now                
            
            channel_found_busy=True
            

            broken_by_lack_of_channels = False

            while node.n_retry and channel_found_busy:
                if profiles[node.profile]["protocol"] == "CAD+Backoff":
                # if noCA_check_busy and (node.cluster!=-2):
                    if profiles[node.profile]["nCAD_DIFS"]>0:
                        no_difs_has_been_found_clear_yet = True
                        while no_difs_has_been_found_clear_yet:
                            for CAD_id in range(profiles[node.profile]["nCAD_DIFS"]):
                                node.n_CAD = node.n_CAD + 1

                                if env.now - node.radio_on_last > node.radio_wake_up_time:
                                    yield env.timeout(node.radio_wake_up_time)
                                    node.radio_wake_up_number +=1


                                ### Wait CAD Duration
                                CAD_time=node.packet.symTime*nCadSym
                                CAD_start_time = env.now

                                on_air_at_CAD_start=start_CAD(node.nodeid) # simulator stores transmitters at CAD start
                                yield env.timeout(CAD_time)
                                channel_found_busy=stop_CAD(node.nodeid,on_air_at_CAD_start) # determines CAD + or -
                                yield env.timeout(node.packet.symTime/2) # process CAD
                                listen_log.append((env.now,CAD_start_time,CAD_time,node.nodeid,node.cycle,int(node.ch)))
                                node.radio_on_last = env.now

                                if channel_found_busy or (CAD_id==(profiles[node.profile]["nCAD_DIFS"]-1)) :
                                    break

                                to_wait_bw_cads = node.packet.symTime*profiles[node.profile]["DIFS_inter_CAD_sym"]

                                if to_wait_bw_cads > node.radio_wake_up_time:
                                    node.radio_on_last = -10000 # force re-wake_up of radio
                                    to_wait_bw_cads -= node.radio_wake_up_time

                                yield env.timeout(to_wait_bw_cads)                            
                                # yield env.timeout(node.packet.symTime*profiles[node.profile]["DIFS_inter_CAD_sym"])

                            if channel_found_busy: # DIFS went busy
                                if profiles[node.profile]["CH_DIFS"]: #TODO: new proto xCH_CAD+BO
                                    if node.update_channel():
                                        yield env.timeout(node.radio_config_time)
                                        node.radio_config_number += 1
                                    else: # no change possible, break anyway, but before, reset avail chan
                                        node.update_available_channels() 
                                        break # while
                                else: # Do single difs, break anyway
                                    break # while                        
                            else: # chan found free
                                break # while


                    else:
                        channel_found_busy=False

                else:
                    channel_found_busy=False
                        
                if channel_found_busy:

                    if profiles[node.profile]["active_BO_when_busy"]: # to wait channel being freed
                        cads_in_busy = 0
                        cads_clear = 0
                        while profiles[node.profile]["active_BO_when_busy_max"]==-1 or cads_in_busy<profiles[node.profile]["active_BO_when_busy_max"]:
                            node.n_CAD = node.n_CAD + 1

                            if env.now - node.radio_on_last > node.radio_wake_up_time:
                                yield env.timeout(node.radio_wake_up_time)
                                node.radio_wake_up_number +=1

                            ### Wait CAD Duration
                            CAD_time=node.packet.symTime*nCadSym
                            CAD_start_time = env.now


                            on_air_at_CAD_start=start_CAD(node.nodeid) # simulator stores transmitters at CAD start
                            yield env.timeout(CAD_time)
                            channel_found_busy=stop_CAD(node.nodeid,on_air_at_CAD_start) # determines CAD + or -
                            yield env.timeout(node.packet.symTime/2) # process CAD
                            listen_log.append((env.now,CAD_start_time,CAD_time,node.nodeid,node.cycle,int(node.ch)))
                            node.radio_on_last = env.now                                

                            if channel_found_busy:
                                # node.n_retry = node.n_retry - 1
                                node.I_heard_activity=True
                                cads_in_busy += 1
                                cads_clear = 0
                            else:
                                cads_clear += 1

                                if cads_clear == profiles[node.profile]["nCAD_DIFS"]: 
                                    break
                                

                            to_wait_bw_cads = node.packet.symTime*profiles[node.profile]["active_BO_when_busy_iCs"]

                            if to_wait_bw_cads > node.radio_wake_up_time:
                                node.radio_on_last = -10000 # force re-wake_up of radio
                                to_wait_bw_cads -= node.radio_wake_up_time

                            yield env.timeout(to_wait_bw_cads)


                    if channel_found_busy:

                        node.n_retry = node.n_retry - 1


                        # if passive_BO_when_busy and (node.n_retry>1 or (not abort_after_retries)):# don't BO if you're gonna abort anyway
                        if profiles[node.profile]["passive_BO_when_busy"] and (node.n_retry>0):# don't do busy BO at the end, do ultimate BO
                            #here we just delay by a random backoff timer to retry again
                            #random backoff [Wbusy_min_airtime_B,2**Wbusy_BE]
                            node.backoff = airtime(node.packet.sf,node.packet.cr,profiles[node.profile]["Wbusy_min_airtime_B"],node.packet.bw,explicit=(LoRa_PHY_HDR==0)) + node.packet.symTime*rng.integers(0,2**node.Wbusy_BE+1)
                            if profiles[node.profile]["Wbusy_exp_backoff"]:
                                if node.Wbusy_BE<profiles[node.profile]["Wbusy_maxBE"]:
                                    node.Wbusy_BE=node.Wbusy_BE + 1
                            if profiles[node.profile]["Wbusy_add_max_toa"]:            
                                yield env.timeout(airtime(node.packet.sf,node.packet.cr,max_payload_size,node.packet.bw,explicit=(LoRa_PHY_HDR==0))+node.backoff)
                            else:
                                yield env.timeout(node.backoff)
                            node.backoff=0

                if not channel_found_busy:
                    if profiles[node.profile]["protocol"] == "CAD+Backoff" and profiles[node.profile]["BO_when_clear"]:
                        
                        if profiles[node.profile]["active_clear_BO"]: #perform CADs during BO
                            if node.remainder == -1:

                                # node.backoff=rng.integers(Wclear_min_sym,2**node.Wclear_BE+1)                        
                                node.remainder=rng.integers(profiles[node.profile]["Wclear_min_CAD"],profiles[node.profile]["Wclear_max_CAD"]+1)                        

                            for CAD_id in range(node.remainder):
                                node.n_CAD = node.n_CAD + 1

                                if env.now - node.radio_on_last > node.radio_wake_up_time:
                                    yield env.timeout(node.radio_wake_up_time)
                                    node.radio_wake_up_number +=1

                                ### Wait CAD Duration
                                CAD_time=node.packet.symTime*nCadSym
                                CAD_start_time = env.now


                                on_air_at_CAD_start=start_CAD(node.nodeid) # simulator stores transmitters at CAD start
                                yield env.timeout(CAD_time)
                                channel_found_busy=stop_CAD(node.nodeid,on_air_at_CAD_start) # determines CAD + or -
                                yield env.timeout(node.packet.symTime/2) # process CAD
                                listen_log.append((env.now,CAD_start_time,CAD_time,node.nodeid,node.cycle,int(node.ch)))
                                node.radio_on_last = env.now                                

                                if channel_found_busy:
                                    node.n_retry = node.n_retry - 1
                                    break

                                if (CAD_id==(node.remainder-1)) : # all clear, go transmit
                                    break

                                to_wait_bw_cads = node.packet.symTime*profiles[node.profile]["BO_inter_CAD_sym"]

                                if to_wait_bw_cads > node.radio_wake_up_time:
                                    node.radio_on_last = -10000 # force re-wake_up of radio
                                    to_wait_bw_cads -= node.radio_wake_up_time

                                yield env.timeout(to_wait_bw_cads)

                            if channel_found_busy:
                                # node.remainder = node.remainder-1-CAD_id
                                node.remainder = node.remainder-CAD_id # last was busy, we need remainder clears to exit BO
                                # node.remainder = max(1,node.remainder-1-CAD_id)
                            else: # BO is successfully finished
                                node.remainder = -1
                        else:
                            node.backoff=node.packet.symTime*rng.integers(profiles[node.profile]["Wclear_min_sym"],profiles[node.profile]["Wclear_max_sym"]+1)
                            yield env.timeout(node.backoff)
                            node.backoff=0


                            if profiles[node.profile]["nCAD_DIFS"]>0:
                                for CAD_id in range(profiles[node.profile]["nCAD_DIFS"]):
                                    node.n_CAD = node.n_CAD + 1

                                    if env.now - node.radio_on_last > node.radio_wake_up_time:
                                        yield env.timeout(node.radio_wake_up_time)
                                        node.radio_wake_up_number +=1

                                    ### Wait CAD Duration
                                    CAD_time=node.packet.symTime*nCadSym
                                    CAD_start_time = env.now

                                    on_air_at_CAD_start=start_CAD(node.nodeid) # simulator stores transmitters at CAD start
                                    yield env.timeout(CAD_time)
                                    channel_found_busy=stop_CAD(node.nodeid,on_air_at_CAD_start) # determines CAD + or -
                                    yield env.timeout(node.packet.symTime/2) # process CAD
                                    listen_log.append((env.now,CAD_start_time,CAD_time,node.nodeid,node.cycle,int(node.ch)))
                                    node.radio_on_last = env.now                                

                                    if channel_found_busy:
                                        node.n_retry = node.n_retry - 1
                                        break

                                    if (CAD_id==(profiles[node.profile]["nCAD_DIFS"]-1)) :
                                        break
                                    
                                    to_wait_bw_cads = node.packet.symTime*profiles[node.profile]["DIFS_inter_CAD_sym"]

                                    if to_wait_bw_cads > node.radio_wake_up_time:
                                        node.radio_on_last = -10000 # force re-wake_up of radio
                                        to_wait_bw_cads -= node.radio_wake_up_time

                                    yield env.timeout(to_wait_bw_cads)

                            else:
                                channel_found_busy=False

                if channel_found_busy:
                    if profiles[node.profile]["CA_Change_Channel"]:
                        if node.n_retry > 0:
                            if node.update_channel():
                                yield env.timeout(node.radio_config_time)
                                node.radio_config_number += 1
                            else:
                                broken_by_lack_of_channels = True
                                break # go aloha


            # end of big while
            node.update_available_channels()
            node.Wbusy_BE=profiles[node.profile]["Wbusy_BE"] if "Wbusy_BE" in profiles[node.profile] else -1
            # node.backoff = 0
            node.total_retry += profiles[node.profile]["n_retry"] - node.n_retry if "n_retry" in profiles[node.profile] else 0
            node.remainder = -1
            

            # exited while without transmiting => abort or ALOHA
            if (node.n_retry==0 or broken_by_lack_of_channels ) and profiles[node.profile]["abort_after_retries"]:
                node.n_aborted = node.n_aborted +1
                # node.n_retry=profiles[node.profile]["n_retry"]
                # node.Wbusy_BE=profiles[node.profile]["Wbusy_BE"] if "Wbusy_BE" in profiles[node.profile] else -1
            else:
                if (node.n_retry==0 or broken_by_lack_of_channels ): # aloha after retries
                    # node.backoff=rng.integers(profiles[node.profile]["CA_ultim_backoff_min"],profiles[node.profile]["CA_ultim_backoff_max"]+1)# 64 == 2**Wbusy_maxBE
                    node.backoff=node.packet.symTime*rng.integers(profiles[node.profile]["Wultim_min_sym"],profiles[node.profile]["Wultim_max_sym"]+1)
                    yield env.timeout(node.backoff)   
                    node.backoff=0
                    node.kamikaze += 1


                if env.now - node.radio_on_last > node.radio_wake_up_time:
                    yield env.timeout(node.radio_wake_up_time)
                    node.radio_wake_up_number +=1

                node.n_data_sent = node.n_data_sent + 1
                node.n_payload_sent += node.packet.dataPayloadSize
                if node.cluster!=-2:
                    nrSent+=1
                # node.total_retry += profiles[node.profile]["n_retry"] - node.n_retry if "n_retry" in profiles[node.profile] else 0
                # node.latency = node.latency + (env.now-node.want_transmit_time)
                node.latency = node.latency + (env.now-node.queued_time)
                if (node.nodeid in packetsOnAir):
                    print("ERROR: DATA packet already in",file=sys.stderr)
                # else:

                heard_at_gateways = (node.packet.rss_GWs - node.packet.noise_dB > get_sensitivity(node.packet.sf,node.packet.bw))
                if not heard_at_gateways.any(): # no gw receive it, too far

                # if node.packet.rssi < get_sensitivity(node.packet.sf,node.packet.bw):
                    # print("node {}: DATA packet will be lost".format(node.nodeid))
                    node.packet.lost = True
                    if full_distances:#on air anyway
                        checkcollision(node.packet,heard_at_gateways)
                        packetsOnAir.append(node.nodeid)
                        node.packet.addTime = env.now
                else:
                    node.packet.lost = False
                    # check collision at GW / local impacts
                    checkcollision(node.packet,heard_at_gateways)
                    packetsOnAir.append(node.nodeid)
                    node.packet.addTime = env.now
                if log_events:
                    MainLogger.info((node.nodeid,"TX_start",env.now))


                channel_busy_data[node.nodeid]=True
                channel_log.append((env.now,node.packet.rectime,node.nodeid,node.cycle,int(node.ch)))
                yield env.timeout(node.packet.rectime)
                channel_busy_data[node.nodeid]=False
                if log_events:
                    MainLogger.info((node.nodeid,"TX_stop",env.now))

                node.radio_on_last = env.now                                
        
                if node.packet.lost:
                    node.packet.collided=0
                    if node.cluster!=-2:
                        nrLost += 1
                    node.n_lost+=1
                else: # check if received and not collided somewhere
                    node.packet.collided=1
                    for gw_id in range(len(heard_at_gateways)):
                        if heard_at_gateways[gw_id]==1 and node.packet.collided_at_GWs[gw_id]==0:
                            node.packet.collided=0
                            break                    
                if node.packet.collided == 1:
                    if node.cluster!=-2:
                        nrCollisions = nrCollisions + 1
                    node.n_collided+=1
                if node.packet.collided == 0 and not node.packet.lost:
                    if node.cluster!=-2:
                        nrReceived = nrReceived + 1
                    # current_latency=env.now-node.want_transmit_time
                    current_latency=env.now-node.queued_time
                    node.min_success_latency = min(node.min_success_latency, current_latency)
                    node.success_latency = node.success_latency + current_latency
                    node.n_data_success+=1
                    node.n_payload_success += node.packet.dataPayloadSize
                    # print("node {} {}: DATA packet has been correctly transmitted".format(node.nodeid, env.now))
                if node.packet.processed == 1:
                    if node.cluster!=-2:
                        nrProcessed = nrProcessed + 1
            
                # complete packet has been received by base station
                # can remove it
                if (node.nodeid in packetsOnAir):
                    packetsOnAir.remove(node.nodeid)
                # reset the packet
                node.packet.stop_interfering()
                node.packet.collided = 0
                node.packet.collided_at_GWs = [0]*len(heard_at_gateways)
                node.packet.processed = 0
                node.packet.lost = False

                node.packet.repropagate()     
                
                # node.update_available_channels()
                node.last_tx_ch = node.ch
                if node.update_channel():
                    yield env.timeout(node.radio_config_time)
                    node.radio_config_number +=1
                # node.n_retry=profiles[node.profile]["n_retry"] if "n_retry" in profiles[node.profile] else -1
                # node.Wbusy_BE=profiles[node.profile]["Wbusy_BE"] if "Wbusy_BE" in profiles[node.profile] else -1
                # node.backoff = 0

            node.n_retry=profiles[node.profile]["n_retry"] if "n_retry" in profiles[node.profile] else -1

        ##################################
        ######## Checking end of Simu ####
        ##################################
        if nrScheduled > targetSchedPacket:
            endSim=env.now
            ## ! ##
            return 


# insert parameters into loop
def main_with_params(params):
    #############################################
    # Parameters
    #############################################

        #############################################
        # Parameters for this scenario
        #############################################


            ######### Channel properties ################
    #CAD reliability probability
    #set to 0 to always assume that CAD indicates a free channel so that CA will be always used, or transmit immediately in ALOHA
    #set to 100 for a fully reliable CAD, normally there should not be collision at all
    #set to [1,99] to indicate a reliability percentage: i.e. (100-CAD_prob) is the probability that CAD reports a free channel while channel is busy            
    global CAD_prob
    global var_CAD_prob     # set CAD_prob variable according to distance and path loss: at 400m, uniform 20%; at -133.25dBm, 100% 
    global CAD_diplos       # array representing CAD true positive prob of success as a function of int distances (m)
    global full_distances   # set rssi model based on distance tx-rx, as opposed to default model (based on dist. tx-gw)
    global lora24GHz        # set sensitivity for this band #set to True if LoRa 2.4Ghz is considered
    global full_collision   # do the full collision check (time overlap and capture, not just sf+bw at a given point in time)
    global gaussian_noise   # if set true each packet will be applied a gaussian noise of fixed mean (see constants) toward any receptor if $full_distances$ otherwise toward GW; otherwise no noise applied.  
    global capture_interf_lock_coef # how impact of interference is decreased after min preamble symbols heard
    global gamma_ED         # Path Loss Exponent for ED-ED links, default in constants.py
    global gamma_GW         # Path Loss Exponent for ED-GW links, default in constants.py
    global normal_gamma_ED  # if True (default), ED-ED PLEs are normally distributed (mean= gamma_ED, stdev sigma_gamma_ED) 
    global sigma_gamma_ED   # Standard deviation of (normally distributed) ED-ED PLEs    
    global intra_cluster_gamma_gain # inside clusters, the PLEs are improved (reduced) by this gain
    global CAD_gamma_ED_delta         # Path Loss Exponent delta for CAD on ED-ED links, default in constants.py
    global CAD_normal_gamma_ED  # if True (default), ED-ED CAD PLEs are normally distributed (mean= CAD_gamma_ED, stdev CAD_sigma_gamma_ED) 
    global CAD_sigma_gamma_ED   # Standard deviation of (normally distributed) ED-ED CAD PLEs

    global buildings_km     # Density of buildings per km of distance

    global GW_sensitivity_gain # GW are more sensible than devices => delta in dB
    global radio_config_time #5 #ms
    global radio_wake_up_time #5 #ms


            ######### Distribution&Traffic properties ################
    global nrNodes          # number of devices
    global externals_prop   # proportion of non clustered devices that are external ones with aloha traffic (no collision avoidance)

    global profiles         # possible node profiles  
    global node_profiles    # distribution of node profiles in this instance

    global maxBSReceives    # maximum number of packets the BSs can receive at the same time, not in use here currently
    global avgSendTime      # average interval between packet arrival
    global packetLength     # fixed size of packets (#the selected packet length)
    global variablePayloadSize # if True, (uniform) variation of payload in e.g. (40,100) octet 
    global normalPayloadSize # if True, normal variation of payload in e.g. (clip_min=0,clip_max=255 or max_payload_size,mu=60,sigma=15) octet 
    global dist_min_payload_size # minimum payload size in a distribution.
    global dist_max_payload_size # maximum payload size in a distribution.
    global normaldist_mean_payload_size # mean payload size in a normal distribution.
    global normaldist_sigma_payload_size # std dev sigma of payload size in a normal distribution.
    global max_payload_size # maximum of allowed payload size, have impact on CAS21 NAV period. #LoRa Phy has a maximum of 255N but you can decide for smaller max value (as with SigFox for instance) (can be different from dist, it's whats expected by protocol)
    global targetSentPacket # number of packets tried in simulation. nb of packet to be sent per node. #targetSentPacket*nrNodes will be the target total number of sent packets before we exit simulation
    global targetSchedPacket # number of packets tried in simulation. nb of packet to be sent per node. #targetSentPacket*nrNodes will be the target total number of scheduled packets before we exit simulation    
    global distribType      # type of traffic (#the selected distribution)
    global shuffle_start    # add a random uniform node.period before starting

            ######### Simulation properties ################
    # experiments: (obsolete) (expe 4 is used in this version 2025-04-10)
    # 0: packet with longest airtime, aloha-style experiment
    # 1: one with 3 frequencies, 1 with 1 frequency
    # 2: with shortest packets, still aloha-style
    # 3: with shortest possible packets depending on distance
    # 4: THIS CURRENT EXPERIMENT
    # "SF7BW500CAD4": first ever "named" experiment" in simu
    global experiment
    global exp4SF #SF value for experiment 4 (sf12)
    global minsensi                 # obsolete
    global Ptx                    # static default device tx power 
    global transmit_at_adjusted_power  # if True, devices will use txpow adjusted to reach at least one GW (the easiest) (among constants.power_levels and at most Ptx)  
    global max_queue_size   # max frames queued
    global nCadSym          # number of Symbols for CAD. NB: in DS_SX1261-2_V2_1 p40, Semtech mentions half a symbol to process CAD. Here assumed process made in parallel with next task. 
    global detection_min_preamb_symb # minimum # of preamble symbols needed to be heard to detect a preamble
    # global Interrupts_on_header_valid # set to true if PHY is able to stop RX mode after HeaderValid interrupt   
    global LoRa_PHY_HDR # GG: explicit header (H=0) or implicit header (H=1) for data frames
    global rayleigh_fading      # if set true, adds a rayleigh distributed dB value corrected to mean 0 
    global rayleigh_mean_dB        # mean of uncorrected rayleigh distribution. np.sqrt(2 / np.pi)*mean is then the "scale" or "mode" parameter.
    global locally_obstructed       # if set true, adds a normal variable for local obstruction in aperture of antenna
    global buildings_attenuated     # if set true, adds an attenuation linear with distance and number of buildings on links

    # global simtime
    global MainLogger,log_events # a node level event logger, very verbose !!!! set to False by default, around (2e6 lines, 100MB)/simu
    global keep_chan_log # same idea, but this boolean choose if you keep it in disk (in res dict), otherwise weeped out of RAM at end of simu
    global keep_Global_TT_IGTs # also a huge list to be kept or not, the network inter gen time 
            ######### Topological properties ################
    # max distance: 300m in city, 3000 m outside (5 km Utz experiment)
    # also more unit-disc like according to Utz
    global maxDist          # max dist GW device considered when building topology. defaults to sensitivity threshold.
    global distance_matrix  # numpy array with distances between devs
    # base station placement
    global center_x              # topo center x coord
    global center_y              # topo center y coord
    global xmax             # for old plots
    global ymax             # for old plots

    global gateways         # dict listing gateways


    #############################################
    # Variables
    #############################################
            ######### Channel State  Vars ####################
    #indicate whether channel is busy or not, we differentiate between channel_busy_rts and channel_busy_data
    #to get more detailed statistics
    global channel_busy_rts
    global channel_busy_data
    global packetsOnAir
    global channel_log
    global listen_log
            ######### Simu monitoring Vars ####################
    global lastDisplayTime  # print nprocessed packets only every 10000 and store time 
            ######### Simu stats on inter-transmit time Vars ####################
    global n_transmit
    global inter_transmit_time
    global max_inter_transmit_time
    global last_transmit_time
            ######### Simu control Vars ####################
    global env
    global endSim           # keep the end simultion time, at which we reach targetSentPacket
            ######### Simu components Vars ####################
    global nodes
            ######### Simu Stats Vars ####################
    global nrCollisions # one frame is not collided if at least one gw received it 
    global nrCaptures # one frame is captured at most once on each GW 
    global nrRTSCollisions
    global nrReceived
    global nrRTSReceived
    global nrProcessed
    global nrSent
    global nrRTSProcessed
    global nrLost
    global nrRTSLost
    global nrScheduled      # generated 
    global powerCaptures

            ######### Ideal Mechanism Vars ####################
    global ideal_latest_start,ideal_latest_time


    ######################################################################################
    ######################################################################################
    ######################################################################################
    ######################################################################################
    ######################################################################################
    ######################################################################################
    ######################################################################################
    ###### C'est Parti ! Let's go! ¡Empezamos!
    ######################################################################################
    ######################################################################################
    ######################################################################################
    ######################################################################################
    ######################################################################################
    ######################################################################################
    ######################################################################################


    #############################################
    # Parameters
    #############################################
        #############################################
        # Parameters for this scenario
        #############################################

            ######### Channel properties ################
    CAD_prob = params["CAD_prob"] if "CAD_prob" in params else 50
    var_CAD_prob= params["var_CAD_prob"] if "var_CAD_prob" in params else False
    full_distances = params["full_distances"] if "full_distances" in params else False
    lora24GHz = False
    full_collision = params["full_collision"] if "full_collision" in params else True
    gaussian_noise = params["gaussian_noise"] if "gaussian_noise" in params else False

    capture_interf_lock_coef = params["capture_interf_lock_coef"] if "capture_interf_lock_coef" in params else .1#6# dB
    gamma_ED = params["gamma_ED"] if "gamma_ED" in params else constants.gamma
    gamma_GW = params["gamma_GW"] if "gamma_GW" in params else constants.gamma_GW

    normal_gamma_ED = params["normal_gamma_ED"] if "normal_gamma_ED" in params else True
    sigma_gamma_ED = params["sigma_gamma_ED"] if "sigma_gamma_ED" in params else constants.sigma_gamma_ED

    CAD_gamma_ED_delta = params["CAD_gamma_ED_delta"] if "CAD_gamma_ED_delta" in params else constants.CAD_gamma_ED_delta

    CAD_normal_gamma_ED = params["CAD_normal_gamma_ED"] if "CAD_normal_gamma_ED" in params else True
    CAD_sigma_gamma_ED = params["CAD_sigma_gamma_ED"] if "CAD_sigma_gamma_ED" in params else constants.CAD_sigma_gamma_ED


    intra_cluster_gamma_gain = params["intra_cluster_gamma_gain"] if "intra_cluster_gamma_gain" in params else 0.
    buildings_km = params["buildings_km"] if "buildings_km" in params else 0.5

    GW_sensitivity_gain = params["GW_sensitivity_gain"] if "GW_sensitivity_gain" in params else 5 #dB

            ######### Distribution&Traffic properties ################
    nrNodes = params["nrNodes"]
    nrGWs = params["nrGWs"]
    externals_prop = params["externals_prop"] if "externals_prop" in params else 0.


    node_profiles = params["node_profiles"]
    profiles = params["profiles"]

    radio_config_time = params["radio_config_time"] if "radio_config_time" in params else 5 #ms
    radio_wake_up_time = params["radio_wake_up_time"] if "radio_wake_up_time" in params else 5 #ms

    maxBSReceives = 8
    avgSendTime = params["avgSendTime"]
    packetLength = 104
    variablePayloadSize = params["variablePayloadSize"] if "variablePayloadSize" in params else False
    normalPayloadSize = params["normalPayloadSize"] if "normalPayloadSize" in params else False
    dist_min_payload_size = params["dist_min_payload_size"] if "dist_min_payload_size" in params else 40
    dist_max_payload_size = params["dist_max_payload_size"] if "dist_max_payload_size" in params else 100
    normaldist_mean_payload_size = params["normaldist_mean_payload_size"] if "normaldist_mean_payload_size" in params else 60
    normaldist_sigma_payload_size = params["normaldist_sigma_payload_size"] if "normaldist_sigma_payload_size" in params else 15

    max_payload_size = params["max_payload_size"] if "max_payload_size" in params else 150
    targetSentPacket = 2000
    targetSentPacket = targetSentPacket * nrNodes
    targetSchedPacket = params["targetSchedPacket"] if "targetSchedPacket" in params else 1000
    # targetSchedPacket *= nrNodes
    #### Later => we need to exclude interferers


    # distribType=uniformDistribType
    # distribType=expoDistribType
    # distribType=perioDistribType
    distribType=expoDistribType if params["distrib"]=="expo" else uniformDistribType if params["distrib"]=="unif" else perioDistribType
    shuffle_start = params["shuffle_start"] if "shuffle_start" in params else False
            ######### Simulation properties ################
    experiment = params["experiment"]
    exp4SF=12

    # get_sensitivity(node.packet.sf,node.packet.bw) but: 
        # minsensi was useful before existence/independance of topo_builder.py
        # kept for compatibility experiment 3, 5, etc.
    if lora24GHz:
        Ptx=constants.Ptx_2dot4GHz
        if experiment in [0,1,4,6,7]:
            minsensi = constants.sensi_2dot4GHz[7,2]     # 7th row is SF12, 2nd column is BW203
        elif experiment == 2:
            minsensi = constants.sensi_2dot4GHz[0,5]     # row 0 is SF5, 5th column is BW1625
        elif experiment in [3,5]:
            minsensi = np.amin(constants.sensi_2dot4GHz) ## Experiment 3 can use any setting, so take minimum
    else:
        Ptx=constants.Ptx_subGHz
        if experiment in [0,1,4,6,7]:
            minsensi = constants.sensi_subGHz[6,2]     # 6th row is SF12, 2nd column is BW125
        elif experiment == 2:
            minsensi = constants.sensi_subGHz[0,3]     # first row is SF6, 4th column is BW500
        elif experiment in [3,5]:
            minsensi = np.amin(constants.sensi_subGHz) ## Experiment 3 can use any setting, so take minimum
        elif experiment =="SF7BW500CAD4":
            minsensi = constants.sensi_subGHz[1,3]     # second row is SF7, 4th column is BW500    

    transmit_at_adjusted_power = params["transmit_at_adjusted_power"] if "transmit_at_adjusted_power" in params else False
    
    max_queue_size = params["max_queue_size"] if "max_queue_size" in params else 8


    nCadSym = params["nCadSym"] if "nCadSym" in params else 4
    detection_min_preamb_symb = params["detection_min_preamb_symb"] if "detection_min_preamb_symb" in params else 3


    LoRa_PHY_HDR = params["LoRa_PHY_HDR"] if "LoRa_PHY_HDR" in params else 0             # GG: explicit header (H=0) or implicit header (H=1)

    rayleigh_fading = params["rayleigh_fading"] if "rayleigh_fading" in params else False
    rayleigh_mean_dB = params["rayleigh_mean_dB"] if "rayleigh_mean_dB" in params else 0

    locally_obstructed = params["locally_obstructed"] if "locally_obstructed" in params else False
    buildings_attenuated = params["buildings_attenuated"] if "buildings_attenuated" in params else False

    # simtime = params["simtime"]
    log_events=params["log_events"]
    keep_chan_log = params["keep_chan_log"] if "keep_chan_log" in params else False
    if log_events:
        MainLogger = load_main_logger(params["start_time"])
        MainLogger.info("Started")
        MainLogger.info('start_time:{0}'.format(params["start_time"]))

    keep_Global_TT_IGTs = params["keep_Global_TT_IGTs"] if "keep_Global_TT_IGTs" in params else False

    this_topo=pickle.load(open('results/{0}_topos.dat'.format(params["start_time"]), 'rb'))[params["topo"]]
    maxDist=this_topo['maxDist']*params["topo_scale"]
    distance_matrix=build_dist_mat(this_topo)*params["topo_scale"]

    #if no full distance with multiGW: mono eval of CAD at center of topo; besides, eval gw collisions as with full dist, no dev collision eval  
    center_x = this_topo['center']['center_x']*params["topo_scale"]
    center_y = this_topo['center']['center_y']*params["topo_scale"]
    xmax = center_x + this_topo['maxDist']*params["topo_scale"] + 20*params["topo_scale"]
    ymax = center_y + this_topo['maxDist']*params["topo_scale"] + 20*params["topo_scale"]


    #############################################
    # Variables
    #############################################
            ######### Channel State  Vars ####################
    channel_busy_rts = [False]*nrNodes
    channel_busy_data = [False]*nrNodes
    packetsOnAir = []
    channel_log = []
    listen_log = []
            ######### Simu monitoring Vars ####################
    lastDisplayTime=-1
            ######### Simu stats on inter-transmit time Vars ####################
    n_transmit = 0
    inter_transmit_time = 0
    max_inter_transmit_time = 40

    last_transmit_time = 0
            ######### Simu control Vars ####################
    env = simpy.Environment()
    endSim=0
            ######### Simu components Vars ####################
    nodes = []
            ######### Simu Stats Vars ####################
    nrCollisions = 0
    nrCaptures = 0
    nrRTSCollisions = 0
    nrReceived = 0
    nrRTSReceived = 0
    nrProcessed = 0
    nrSent = 0
    nrRTSProcessed = 0
    nrLost = 0
    nrRTSLost = 0
    nrScheduled = 0
    powerCaptures = []
            ######### Ideal Mechanism Vars ####################
    ideal_latest_start = [0]*len(constants.Channel_list)
    ideal_latest_time = [0]*len(constants.Channel_list)



    ###########
    ###########



    if experiment==6:
        nrNodes=9

    if experiment==7:
        nrNodes=5



    ###########
    # adjust clustering
    ###########


    nonclusts=[]
    for n in range(nrNodes):
        if this_topo['nodes'][n]['cluster'] == -1:
            nonclusts.append(n)

    to_externalize=round(len(nonclusts)*externals_prop)
    for ext_id in range(to_externalize):
        this_topo['nodes'][nonclusts[ext_id]]['cluster'] = -2


    # each node has cluster, let's profile them all
    # first recount
    clusters_recount={}
    for n in range(nrNodes):
        if this_topo['nodes'][n]['cluster'] in clusters_recount:
            clusters_recount[this_topo['nodes'][n]['cluster']].append(n)
        else:
            clusters_recount[this_topo['nodes'][n]['cluster']]=[n]

    contributors=nrNodes
    if -2 in clusters_recount: # externals
        population = len(clusters_recount[-2])
        contributors -= population
        subgroups = []
        subgroups_names = []
        for prof_distrib in node_profiles["externals"]["distrib"]:
            subgroups.append(round(node_profiles["externals"]["distrib"][prof_distrib]*population))
            subgroups_names.append(prof_distrib)
        turno = 0
        while sum(subgroups)<population:
            subgroups[turno]+=1
            turno+=1
            turno%=len(subgroups)


        it = iter(clusters_recount[-2])
        populated_subgroups = [[next(it) for _ in range(size)] for size in subgroups]

        for popsub_id in range(len(populated_subgroups)):
            popsub = populated_subgroups[popsub_id]
            prof_to_apply = subgroups_names[popsub_id]
            for nid in popsub:
                this_topo['nodes'][nid]['profile'] = prof_to_apply
    

    ######### packets globally scheduled are scheduled by legit contributors
    targetSchedPacket *= contributors



    if -1 in clusters_recount: # unclustered
        population = len(clusters_recount[-1])
        subgroups = []
        subgroups_names = []
        for prof_distrib in node_profiles["unclustered"]["distrib"]:
            subgroups.append(round(node_profiles["unclustered"]["distrib"][prof_distrib]*population))
            subgroups_names.append(prof_distrib)
        turno = 0
        while sum(subgroups)<population:
            subgroups[turno]+=1
            turno+=1
            turno%=len(subgroups)


        it = iter(clusters_recount[-1])
        populated_subgroups = [[next(it) for _ in range(size)] for size in subgroups]

        for popsub_id in range(len(populated_subgroups)):
            popsub = populated_subgroups[popsub_id]
            prof_to_apply = subgroups_names[popsub_id]
            for nid in popsub:
                this_topo['nodes'][nid]['profile'] = prof_to_apply

    nb_clusters = 0
    nb_clustered = 0
    clust_ids = []
    for clust_id in clusters_recount:
        if clust_id in [-2,-1]:
            continue
        nb_clusters +=1
        nb_clustered += len(clusters_recount[clust_id])
        clust_ids.append(clust_id)


    if nb_clusters>0:
        if node_profiles["clustered"]["distribType"] == "full_cluster":

            subgroups = []
            subgroups_names = []
            for prof_distrib in node_profiles["clustered"]["distrib"]:
                subgroups.append(round(node_profiles["clustered"]["distrib"][prof_distrib]*nb_clusters))
                subgroups_names.append(prof_distrib)
            turno = 0
            while sum(subgroups)<nb_clusters:
                subgroups[turno]+=1
                turno+=1
                turno%=len(subgroups)

            it = iter(clust_ids)
            populated_subgroups = [[next(it) for _ in range(size)] for size in subgroups]


            for popsub_id in range(len(populated_subgroups)):
                popsub = populated_subgroups[popsub_id]
                prof_to_apply = subgroups_names[popsub_id]

                for clust_id in popsub:
                    for nid in clusters_recount[clust_id]:
                        this_topo['nodes'][nid]['profile'] = prof_to_apply

        elif node_profiles["clustered"]["distribType"] == "node_wise":



            population = nb_clustered
            subgroups = []
            subgroups_names = []
            for prof_distrib in node_profiles["clustered"]["distrib"]:
                subgroups.append(round(node_profiles["clustered"]["distrib"][prof_distrib]*population))
                subgroups_names.append(prof_distrib)
            turno = 0
            while sum(subgroups)<population:
                subgroups[turno]+=1
                turno+=1
                turno%=len(subgroups)



            subgroup_id = 0
            
            for nid in range(nrNodes):
            # for nid in this_topo['nodes']:
                

                if this_topo['nodes'][nid]['cluster'] in [-2,-1]:
                    continue
                
                while subgroups[subgroup_id] == 0:
                    subgroup_id +=1 

                prof_to_apply = subgroups_names[subgroup_id]
                this_topo['nodes'][nid]['profile'] = prof_to_apply

                subgroups[subgroup_id] -= 1

    ###########
    ###########
    ## prepare CADs 
    ###########
    ###########


    CAD_diplos = prepare_diplo(constants.loss_per_building_CAD, constants.obstruction_dB_CAD, 12, 125)
    # prepared for sf 12 bw 125
    # in future scenarii with multi sf/ multi bw => prepare a diplo matrix

    ###########
    # numpyfy GWs
    ###########

    gateways = {"xs":[],"ys":[]}
    for gw in this_topo['GWs']:
        gateways["xs"].append(gw["x"]*params["topo_scale"])
        gateways["ys"].append(gw["y"]*params["topo_scale"])
    gateways["xs"]=np.array(gateways["xs"])
    gateways["ys"]=np.array(gateways["ys"])


    ###########
    # Devices
    ###########

    for i in range(0,nrNodes):
        # (self, nodeid, nodex, nodey, node_cluster, profile, nodeType, period, distrib, packetlen):
        node = myNode(
            i, 
            this_topo['nodes'][i]['x']*params["topo_scale"], 
            this_topo['nodes'][i]['y']*params["topo_scale"], 
            this_topo['nodes'][i]['cluster'],
            this_topo['nodes'][i]['profile'],
            endDeviceType, 
            avgSendTime, 
            distribType, 
            packetLength)

        nodes.append(node)

    
    for node in nodes:
        if node.cluster not in [-1,-2]:# cluster -1 is the no-cluster, -2 is the cluster of externals
            for other in nodes:
                if node.cluster==other.cluster:
                    node.packet.gamma_array[other.nodeid]-=intra_cluster_gamma_gain
        node.packet.repropagate(first_time=True)



    # ############################################################### DEBUG
    #     sys.stdout=stdout_print_target
    #     for node in nodes:
    #         print(node.nodeid, node.dist_to_center, node.dist_GWs)
    #     print(0/0)
    # ################################################################


    ###########
    # SIMULATION PROCESS
    ###########
    for node in nodes:
        env.process(communicate(env,node))    
    
    # remove topo from RAM
    this_topo=None
        
    # start simulation
    # env.run(until=simtime)
    env.run()

    if log_events:
        close_logger_handlers(MainLogger)

    #########################
    ### POST SIMU ########
    #########################

    res={}

    # compute energy
    TX = constants.TX 
    RX = constants.RX
    V = constants.V

    # compute energy
    TX = constants.TX_868_14dBm
    RX = constants.RX_boosted_DCDC_mode
    V = constants.V

    MCU_on = constants.MCU_on
    MCU_sleep = constants.MCU_sleep



    #################################################################
    #################################################################
    #################################################################
    #################################################################
    #################################################################
    #################################################################
    #statistic per node
    res["nodes"]={}

    for node in nodes:
        res["nodes"][node.nodeid]={
            "number_of_CAD": node.n_CAD,
            "node_type": 'endDevice' if node.nodeType==endDeviceType else 'relayDevice',
            "node_traffic": 'expo' if node.distrib==expoDistribType else 'uniform',
            # "x":node.x,
            # "y":node.y, 
            "dist_to_center":node.dist_to_center,
            "cluster":node.cluster,
            "profile":node.profile
        }

        #### ENERGY in CAD ####
        str_sf = "SF"+str(node.packet.sf)
        str_bw = "BW"+str(node.packet.bw)
        str_cadsym = str(nCadSym)+"S"

        #consumption must be converted into mA: cad_consumption[node.packet.sf-7]/1e6    
        # energy = (node.packet.symTime * (1/2+nCadSym) * (constants.cad_consumption[str_sf][str_bw][str_cadsym]/3600/1e9) * V * node.n_CAD ) / 1e3
        # corrected: values in nAh for the whole CAD
        energy = ((constants.cad_consumption[str_sf][str_bw][str_cadsym]*3600/1e9) * V * node.n_CAD )
        
        #mike is on during CAD
        energy += (node.packet.symTime * (1/2+nCadSym) * node.n_CAD * MCU_on * V / 1e6)


        node.CAD_energy=energy
        res["nodes"][node.nodeid]["energy_in_CAD_J"]=energy

        ##### TIME in TX 
        time_sending_data=0
        start_sending_data=0
        stop_sending_data=0
        for cl in channel_log:
            if cl[2]==node.nodeid:
                if start_sending_data==0:
                    start_sending_data=cl[0]
                time_sending_data+=cl[1]        
                stop_sending_data=cl[0]+cl[1]        

        # print(time_sending_data,start_sending_data,stop_sending_data, file=sys.stderr)
        # print(0/0)

        #### ENERGY in TX
        energy = (time_sending_data * constants.TX_currents[constants.power_levels.index(node.packet.txpow)] * V) / 1e6


        #mike is on
        energy += (time_sending_data * MCU_on * V) / 1e6

        res["nodes"][node.nodeid]["energy_in_transmission_J"]=energy

        #### ENERGY in RX
        if True:#profiles[node.profile]["protocol"]=="CANL22":
        # if CANL22:
            energy = (node.total_listen_time * RX * V) / 1e6
            #mike is on
            energy += (node.total_listen_time * MCU_on * V) / 1e6

            res["nodes"][node.nodeid]["energy_in_listening_J"]=energy

        #### TOTAL ENERGY
        time_on = time_sending_data + node.total_listen_time + node.packet.symTime * (1/2+nCadSym) * node.n_CAD

        # #mike is on during radio wakeups and configs 
        # energy += (node.radio_wake_up_time * node.radio_wake_up_number * MCU_on * V / 1e6)
        # energy += (node.radio_config_time * node.radio_config_number * MCU_on * V / 1e6)
        time_on += (node.radio_wake_up_time * node.radio_wake_up_number )
        time_on += (node.radio_config_time * node.radio_config_number )


        time_off = endSim - time_on


        res["nodes"][node.nodeid]["radio_config_number"]= node.radio_config_number



        res["nodes"][node.nodeid]["total_energy_J"]=(
            time_sending_data * constants.TX_currents[constants.power_levels.index(node.packet.txpow)] * V + \
            node.total_listen_time * RX * V) / 1e6 + node.CAD_energy + \
            (node.radio_wake_up_time * node.radio_wake_up_number * MCU_on * V / 1e6) + \
            (node.radio_config_time * node.radio_config_number * MCU_on * V / 1e6) + \
            time_off * MCU_sleep * V / 1e6

        #### ENERGY per SUCCESS
        res["nodes"][node.nodeid]["energy_per_success"]=res["nodes"][node.nodeid]["total_energy_J"]/node.n_data_success if node.n_data_success>0 else -1
        res["nodes"][node.nodeid]["length_of_success"]=node.n_payload_success/node.n_data_success if node.n_data_success>0 else -1

        res["nodes"][node.nodeid]["mean_payload_length"]= node.n_payload_gen/node.cycle if node.n_payload_gen>0 else -1
        

        res["nodes"][node.nodeid]["energy_per_payload_success"]= res["nodes"][node.nodeid]["total_energy_J"]/node.n_payload_success if node.n_payload_success>0 else -1


        res["nodes"][node.nodeid]["end_simulation_time"]=" {}ms {}h".format(endSim, float(endSim/3600000))
        res["nodes"][node.nodeid]["cumulated_TX_time_s"]=time_sending_data/1000
        res["nodes"][node.nodeid]["duty_cycle"]=time_sending_data/(stop_sending_data-start_sending_data) if stop_sending_data>start_sending_data else 0
        # if CANL22:
        if True:#profiles[node.profile]["protocol"]=="CANL22":
            res["nodes"][node.nodeid]["cumulated_RX_time_s"]=node.total_listen_time/1000

        # res["nodes"][node.nodeid]["number_of_CAD"]= node.n_CAD #sum (n.n_CAD for n in nodes)
        res["nodes"][node.nodeid]["sent_data_packets"]= node.n_data_sent
        res["nodes"][node.nodeid]["success_data_packets"]= node.n_data_success
        res["nodes"][node.nodeid]["DER"]= node.n_data_success/node.n_data_sent if node.n_data_sent>0 else -1
        res["nodes"][node.nodeid]["DER_method_2"]= (node.n_data_sent-node.n_collided)/node.n_data_sent if node.n_data_sent>0 else -1
        res["nodes"][node.nodeid]["PDR"]= node.n_data_success/(node.n_data_sent+node.n_dropped+node.n_aborted) if (node.n_data_sent+node.n_dropped+node.n_aborted > 0 ) else -1

        res["nodes"][node.nodeid]["payload_byte_delivery_ratio"]= node.n_payload_success/node.n_payload_gen if node.n_payload_gen>0 else -1

        res["nodes"][node.nodeid]["mean_latency"]= node.latency/node.n_data_sent if node.n_data_sent>0 else -1
        res["nodes"][node.nodeid]["mean_success_latency"]= node.success_latency/node.n_data_success if node.n_data_success>0 else -1
        res["nodes"][node.nodeid]["min_success_latency"]= node.min_success_latency
        res["nodes"][node.nodeid]["aborted_packets"]= node.n_aborted
        res["nodes"][node.nodeid]["collided_packets"]= node.n_collided
        res["nodes"][node.nodeid]["captured_packets"]= node.n_captured
        res["nodes"][node.nodeid]["lost_packets"]= node.n_lost
        res["nodes"][node.nodeid]["dropped_packets"]= node.n_dropped
        res["nodes"][node.nodeid]["kamikazed_packets"]= node.kamikaze

        res["nodes"][node.nodeid]["mean_retry"]= node.total_retry/node.n_data_sent if node.n_data_sent>0 else -1

        # if CANL22:
        if profiles[node.profile]["protocol"]=="CANL22":
            res["nodes"][node.nodeid]["sent_rts_packets"]=node.n_rts_sent

    res["settings"]={
        "Nodes": nrNodes,
        "externals_prop":externals_prop,
        "AvgSendTime": avgSendTime,
        "Distribution": 'expoDistribType' if distribType==expoDistribType else 'uniformDistribType',
        "Experiment": experiment,
        # "Simtime": simtime,
        "Full Collision": full_collision,
        "Default_TOA DATA": nodes[0].data_rectime,
        "TOA RTS": nodes[0].rts_rectime,
        # "DIFS": nodes[0].packet.Tpream,
        # "n_retry": n_retry,
        "CAD_prob": CAD_prob,
        "Packet length": packetLength,
        # "targetSentPacket": targetSentPacket, 
        "targetSchedPacket": targetSchedPacket, 
        "gaussian_noise":gaussian_noise,
        "CAD_proba":{i:list(CAD_diplos[i]) for i in CAD_diplos} if False else {}
    }


    legit_nodes=[]
    for n in nodes:
        if n.cluster!=-2:
            legit_nodes.append(n)

    legit_nrNodes = len(legit_nodes)


    sent = sum(n.n_data_sent for n in legit_nodes)
    rts_sent = sum(n.n_rts_sent for n in legit_nodes)
    

    res["TOTAL"]={
        "radio_config_number": sum( n.radio_config_number for n in legit_nodes) / legit_nrNodes,
        "energy_in_CAD_J":sum( n.CAD_energy    for n in legit_nodes)  / legit_nrNodes,
        "energy_in_transmission_J":sum( [res["nodes"][n.nodeid]["energy_in_transmission_J"] for n in legit_nodes])  / legit_nrNodes,
        # "energy_in_listening_J":sum( n.total_listen_time * RX * V for n in legit_nodes) / 1e6,
        "energy_in_listening_J":sum( [res["nodes"][n.nodeid]["energy_in_listening_J"] for n in legit_nodes])   / legit_nrNodes,
        "total_energy_J":sum( [res["nodes"][n.nodeid]["total_energy_J"] for n in legit_nodes])  / legit_nrNodes,
        "end_simulation_time":" {}ms {}h".format(endSim, float(endSim/3600000)),

        "cumulated_TX_time_s":sum( [res["nodes"][n.nodeid]["cumulated_TX_time_s"] for n in legit_nodes])  / legit_nrNodes,
        "number_of_CAD":sum (n.n_CAD for n in legit_nodes)  / legit_nrNodes,

        "sent_data_packets": sent / legit_nrNodes,
        # "mean_latency": sum (float(n.latency)/float(n.n_data_sent) for n in legit_nodes) / legit_nrNodes if ( node.n_data_sent>0 and legit_nrNodes > 0 )else -1,
        "min_success_latency": sum (n.min_success_latency for n in legit_nodes) / legit_nrNodes,
        "aborted_packets": sum (n.n_aborted for n in legit_nodes)  / legit_nrNodes,
        "collided_packets": nrCollisions  / legit_nrNodes,
        "captured_packets": nrCaptures / legit_nrNodes,
        "lost_packets": nrLost  / legit_nrNodes,
        "dropped_packets": sum (n.n_dropped for n in legit_nodes)  / legit_nrNodes,
        "kamikazed_packets": sum (n.kamikaze for n in legit_nodes)  / legit_nrNodes,        



        
        "nrCollisions":nrCollisions,
        "nrCaptures":nrCaptures,
        "nrReceived":nrReceived,
        "nrProcessed":nrProcessed,
        "nrSent":nrSent,
        "nrLost":nrLost,
        "nrScheduled":nrScheduled
    }


    # "mean_latency":
    sum_lat=0
    sum_data_sent=0
    legit_having_sent=0
    for n in legit_nodes:
        if n.n_data_sent>0:
            sum_lat+=n.latency
            sum_data_sent+=n.n_data_sent
            legit_having_sent+=1
    if sum_data_sent>0:
        res["TOTAL"]["mean_latency"]=sum_lat/sum_data_sent/legit_having_sent
    else:
        res["TOTAL"]["mean_latency"]=-1    


    # "mean_success_latency":
    sum_suc_lat=0
    sum_data_suc=0
    for n in legit_nodes:
        if n.n_data_success>0:
            sum_suc_lat+=n.success_latency
            sum_data_suc+=n.n_data_success
    if sum_data_suc>0:
        res["TOTAL"]["mean_success_latency"]=sum_suc_lat/sum_data_suc
    else:
        res["TOTAL"]["mean_success_latency"]=-1        


    res["TOTAL"]["energy_per_success"]=sum( [res["nodes"][n.nodeid]["total_energy_J"] for n in legit_nodes])/sum_data_suc if sum_data_suc>0 else -1

    res["TOTAL"]["length_of_success"]=sum(n.n_payload_success for n in legit_nodes)/sum_data_suc if sum_data_suc>0 else -1


    # if CANL22:
    # if profiles[node.profile]["protocol"]=="CANL22":
    if True:
        res["TOTAL"]["cumulated_RX_time_s"]=sum( (n.total_listen_time) for n in legit_nodes)/legit_nrNodes/1000



    # "mean_retries": 
    sum_retries=0
    sum_data_gen=0
    legit_having_gen=0
    for n in legit_nodes:
        if n.cycle>0:
            sum_retries+=n.total_retry
            sum_data_gen+=n.cycle
            legit_having_gen+=1
    if sum_data_gen>0:
        res["TOTAL"]["mean_retry"]=sum_retries/sum_data_gen/legit_having_gen
        res["TOTAL"]["mean_payload_length"]=sum(n.n_payload_gen for n in legit_nodes)/sum_data_gen
    else:
        res["TOTAL"]["mean_retry"]=-1 
        res["TOTAL"]["mean_payload_length"]=-1 
    # res["TOTAL"]["mean_retry"]=sum((float(n.total_retry)/float(n.n_data_sent)) for n in legit_nodes)/legit_nrNodes

    
    # if CANL22:
    if True:
        res["TOTAL"].update({

            "sent_rts_packets":rts_sent,
            "nrRTSCollisions":nrRTSCollisions,
            "RTS_received_packets": nrRTSReceived,
            "RTS_processed_packets": nrRTSProcessed,
            "RTS_lost_packets": nrRTSLost,

        })


    if sent>0:
        # data extraction rate switched to include losses
        der = (sent-nrCollisions)/float(sent)
        res["TOTAL"]["DER_method_2"]=der
        der = (nrReceived)/float(sent)
        res["TOTAL"]["DER"]=der

    res["TOTAL"]["duty_cycle"]=sum(res["nodes"][n.nodeid]["duty_cycle"] for n in legit_nodes)/legit_nrNodes

    res["TOTAL"]["PDR"]= sum(n.n_data_success for n in legit_nodes)/sum(n.n_data_sent+n.n_dropped+n.n_aborted for n in legit_nodes)
    try:
        res["TOTAL"]["payload_byte_delivery_ratio"]= sum(n.n_payload_success for n in legit_nodes)/sum(n.n_payload_gen for n in legit_nodes)
    except:
        res["TOTAL"]["payload_byte_delivery_ratio"]=0

    try:
        res["TOTAL"]["energy_per_payload_success"]= sum( [res["nodes"][n.nodeid]["total_energy_J"] for n in legit_nodes])  / sum(n.n_payload_success for n in legit_nodes)
    except:
        res["TOTAL"]["energy_per_payload_success"] = -1
 
    res["TOTAL"]["n_transmit"]= n_transmit
    res["TOTAL"]["mean_inter_transmit_time_ms"]=inter_transmit_time/float(n_transmit)


    #### Control inter generation times
    TT_gen_times = [0]
    TT_IGTs=[]
    for n in nodes:
        TT_gen_times+=n.gen_times[1:]
        npks_gen=len(n.gen_times)
        IGTs=[n.gen_times[i]-n.gen_times[i-1] for i in range(1,npks_gen)]
        if n.cluster!=-2:
            TT_IGTs+=IGTs
    
        res["nodes"][n.nodeid]["mean_IGT"] = np.mean(IGTs)
        res["nodes"][n.nodeid]["std_dev_IGT"] = np.std(IGTs)

    res["TOTAL"]["mean_IGT"]=np.mean(TT_IGTs)
    res["TOTAL"]["std_dev_IGT"]=np.std(TT_IGTs)

    # TT_gen_times=sorted(
    TT_gen_times.sort()
    npks_gen=len(TT_gen_times)
    Global_TT_IGTs=[TT_gen_times[i]-TT_gen_times[i-1] for i in range(1,npks_gen)]
    if keep_Global_TT_IGTs:
        res["TOTAL"]["Global_TT_IGTs"]=Global_TT_IGTs
    res["TOTAL"]["short_IGTs"]=np.count_nonzero(np.array(Global_TT_IGTs)<1000)/len(Global_TT_IGTs)



    channel_log.sort(key=lambda tup: tup[0])
    # channel_log.append((env.now,node.packet.rectime,node.nodeid,node.cycle,int(node.ch)))

    res["channels"] = {}
    for chan_id in range(len(constants.Channel_list)):
        res["channels"][chan_id]={}

        channel_log_local = [cl for cl in channel_log if cl[-1]==chan_id]

        busy_dur=0
        busy_start=channel_log_local[0][0]
        busy_stop=max([channel_log_local[i][0]+channel_log_local[i][1] for i in range(len(channel_log_local))])
        last_previous_end=channel_log_local[0][0]+channel_log_local[0][1]
        prev_tx_id=0
        for tx_id in range(1,len(channel_log_local)):
            tx=channel_log_local[tx_id]
            prev_tx=channel_log_local[prev_tx_id]
            if (tx[0]<last_previous_end): #starts before end of previous 
                if (tx[0]+tx[1]>last_previous_end) : #ends after end of previous 
                    busy_dur += (tx[0]+tx[1]) - (prev_tx[0]+prev_tx[1])
                    last_previous_end=tx[0]+tx[1]
                    prev_tx_id=tx_id
            else:
                busy_dur += tx[1]
                last_previous_end=tx[0]+tx[1]
                prev_tx_id=tx_id

        res["channels"][chan_id]["channel_occupation"]=busy_dur/(busy_stop-busy_start)

        
        max_idv_chan_occ_time = max([sum([cl[1] for cl in channel_log_local if cl[2]==n.nodeid]) for n in nodes])
        totalsum_idv_chan_occ_time = sum([sum([cl[1] for cl in channel_log_local if cl[2]==n.nodeid]) for n in nodes])
        
        res["channels"][chan_id]["channel_overlap_ratio"]=(totalsum_idv_chan_occ_time-busy_dur)/(totalsum_idv_chan_occ_time-max_idv_chan_occ_time)

    res["TOTAL"]["channel_overlap_ratio"] = np.mean([res["channels"][chan_id]["channel_overlap_ratio"] for chan_id in range(len(constants.Channel_list))])
    res["TOTAL"]["channel_occupation"] = np.mean([res["channels"][chan_id]["channel_occupation"] for chan_id in range(len(constants.Channel_list))])

    if keep_chan_log:
        res["TOTAL"]["chanlog"]=channel_log
        res["TOTAL"]["listenlog"]= listen_log

    powerChecks=0#len(powerCaptures)
    # print(powerChecks, "powerChecks", file=sys.stderr)
    
    nb_caps=0
    sum_in_ears=0
    max_gw_in_ears=0
    sum_in_ears_with_capture=0

    for n in nodes:
        res["nodes"][n.nodeid]["sum_in_ears"]=0
        res["nodes"][n.nodeid]["nb_caps"]=0
        res["nodes"][n.nodeid]["sum_in_ears_with_capture"]=0
        res["nodes"][n.nodeid]["powerChecks"]=0

        res["nodes"][n.nodeid]["max_overlap_degree"]=0
        res["nodes"][n.nodeid]["max_capture_overlap_degree"]=0


    for pcheck in powerCaptures:
        if pcheck[3]=="gw":
            powerChecks+=1
            sum_in_ears+=pcheck[0]
            max_gw_in_ears=max(max_gw_in_ears,pcheck[0])
            if pcheck[1]:
                nb_caps+=1
                sum_in_ears_with_capture+=pcheck[0]

        else:
            res["nodes"][pcheck[2]]["sum_in_ears"]+=pcheck[0]
            res["nodes"][pcheck[2]]["max_overlap_degree"]=max(res["nodes"][pcheck[2]]["max_overlap_degree"],pcheck[0])
            res["nodes"][pcheck[2]]["powerChecks"]+=1
            if pcheck[1]:
                res["nodes"][pcheck[2]]["nb_caps"]+=1
                res["nodes"][pcheck[2]]["sum_in_ears_with_capture"]+=pcheck[0]
                res["nodes"][pcheck[2]]["max_capture_overlap_degree"]=max(res["nodes"][pcheck[2]]["max_capture_overlap_degree"],pcheck[0])
    

    res["TOTAL"]["GW_power_capture_ratio"]=nb_caps/powerChecks if powerChecks>0 else 0
    res["TOTAL"]["GW_overlap_degree"]=sum_in_ears/powerChecks if powerChecks>0 else 0
    res["TOTAL"]["GW_max_overlap_degree"]=int(max_gw_in_ears) # convert numpy.int64 to int for jsonify
    res["TOTAL"]["GW_capture_overlap_degree"]=sum_in_ears_with_capture/nb_caps if nb_caps>0 else 0

    for n in nodes:
        res["nodes"][n.nodeid]["power_capture_ratio"]=res["nodes"][n.nodeid]["nb_caps"]/res["nodes"][n.nodeid]["powerChecks"] if res["nodes"][n.nodeid]["powerChecks"]>0 else 0
        res["nodes"][n.nodeid]["mean_overlap_degree"]=res["nodes"][n.nodeid]["sum_in_ears"]/res["nodes"][n.nodeid]["powerChecks"] if res["nodes"][n.nodeid]["powerChecks"]>0 else 0
        res["nodes"][n.nodeid]["mean_capture_overlap_degree"]=res["nodes"][n.nodeid]["sum_in_ears_with_capture"]/res["nodes"][n.nodeid]["nb_caps"] if res["nodes"][n.nodeid]["nb_caps"]>0 else 0

    res["TOTAL"]["power_capture_ratio"]=sum(res["nodes"][n.nodeid]["power_capture_ratio"] for n in nodes)/nrNodes
    res["TOTAL"]["mean_overlap_degree"]=sum(res["nodes"][n.nodeid]["mean_overlap_degree"] for n in nodes)/nrNodes
    res["TOTAL"]["mean_capture_overlap_degree"]=sum(res["nodes"][n.nodeid]["mean_capture_overlap_degree"] for n in nodes)/nrNodes
    res["TOTAL"]["max_overlap_degree"]=sum(res["nodes"][n.nodeid]["max_overlap_degree"] for n in nodes)/nrNodes
    res["TOTAL"]["max_capture_overlap_degree"]=sum(res["nodes"][n.nodeid]["max_capture_overlap_degree"] for n in nodes)/nrNodes


        
    
    # print("-- END ----------------------------------------------------------------------"    )


    if "pickle_each" not in params or not params["pickle_each"]:
        return res

    pickle.dump(res, open('results/{0}/{0}_{1}_data.dat'.format(params["start_time"],params["run_index"]), 'wb'))
    # res = {}







#
# "main" program as used as "python lora_csma_sim.py"
#
if __name__ == '__main__':

    # module for the generation of the topology
    import topo_builder

    import time
    import json
    JSON_EXPORT = True

    raw_start=time.localtime()
    start_time=time.strftime("%Y-%m-%d-%H-%M-%S", raw_start)
    experiment=4
    # nrNodes=20
    # nrNodes=400
    nrNodes=100
    nrGWs = 5
    log_events=False
    maxDist_dev_gw=260

    targetSchedPacket=10
    avgSendTime=1500000
    avgSendTime=4000

    cluster_params=('manual-uniform',3,250,(5,10))

    topos={}
    # topos[0]=topo_builder.build_topo(nrNodes,nrGWs,experiment,maxDist_dev_gw,cluster_params=cluster_params)
    topos[0]=topo_builder.build_topo(nrNodes,nrGWs,experiment,maxDist_dev_gw)

    pickle.dump(topos, open('results/{0}_topos.dat'.format(start_time), 'wb'))

    # proto = "LoRa_CSMA"
    proto = "xCANL_CAD"
    # proto = "ideal_FIFO"
    # proto = "CANL_2022"
    # proto = "ALOHA"

    params={
        "start_time":start_time,
        "pickle_each":False,
        "run_index":0,
        "topo":0,
        "topo_scale":1,
        "log_events":log_events,
        "keep_chan_log":False,
        "keep_Global_TT_IGTs":False,        

                ######### Channel properties ################
        "CAD_prob": 50,
        "var_CAD_prob": True,
        # "CAD_prob": 100,
        # "var_CAD_prob": False,
        "full_distances": True,
        # lora24GHz = False
        "full_collision": True,
        "gaussian_noise": True,

        "capture_interf_lock_coef": .1,#6,# dB,
        "gamma_ED": constants.gamma,
        "gamma_GW": constants.gamma_GW,

        "normal_gamma_ED": True,
        "sigma_gamma_ED": constants.sigma_gamma_ED,
        "intra_cluster_gamma_gain": 0.2,

        "CAD_gamma_ED_delta": constants.CAD_gamma_ED_delta,
        "CAD_normal_gamma_ED": True,
        "CAD_sigma_gamma_ED": constants.CAD_sigma_gamma_ED,


        "buildings_km": 0.5,

        "GW_sensitivity_gain": 5, #dB,

                ######### Distribution&Traffic properties ################
        "nrNodes":nrNodes,
        "nrGWs":nrGWs,
        "externals_prop":0.3,

        "radio_config_time":10, #ms
        "radio_wake_up_time":10, #ms

        # maxBSReceives = 8
        "avgSendTime":avgSendTime,
        # packetLength = 104
        "variablePayloadSize":True,
        "normalPayloadSize":True,
        "dist_min_payload_size":40,
        "dist_max_payload_size":100,
        "normaldist_mean_payload_size":60,
        "normaldist_sigma_payload_size":15,

        "max_payload_size":150,
        "targetSchedPacket":targetSchedPacket,

        "distrib":"expo",
        
        "shuffle_start":False,

        "experiment": experiment,

        "max_queue_size":1,
        "nCadSym":4, 

        "detection_min_preamb_symb":6,

        "LoRa_PHY_HDR":0,
        
        "rayleigh_fading":True,
        "rayleigh_mean_dB":1,

        "locally_obstructed":True,
        "buildings_attenuated":True,

        "transmit_at_adjusted_power":False,

        ## in the following, non-null proportions in [0,1] must represent at least one device (TODO: failsafe)
        ## watch it with externals_prop value
        "node_profiles" : {
            "clustered":{
                "distribType":"full_cluster", # "node-wise"
                "distrib":{
                    proto:1,
                    # "LoRa_CSMA":0,
                    # "xCANL_CAD":1,
                    # "ideal_FIFO":0,
                    # "CANL_2022":0,
                    # "ALOHA":0,                
                }
            },
            "unclustered":{
                "distrib":{
                    proto:1,
                    # "LoRa_CSMA":0,
                    # "xCANL_CAD":1,
                    # "ideal_FIFO":0,
                    # "CANL_2022":0,
                    # "ALOHA":0,
                }
            },
            "externals":{ #TODO: adapt node stats variable updates for non aloha externals
                "distrib":{
                    # "LoRa_CSMA":0,
                    # "xCANL_CAD": 0,
                    # "ideal_FIFO":0,
                    # "CANL_2022":0,
                    "ALOHA":1,
                }

            }
        },



        "profiles":{
            "LoRa_CSMA":{
                "protocol":"CAD+Backoff",
                "n_retry":6,
                "abort_after_retries":False,
                "Wultim_min_sym": 0,#66,
                "Wultim_max_sym": 0,#132,     
                "Wclear_min_CAD":2,           
                "Wclear_max_CAD":22,           
                "nCAD_DIFS":2,      
                "DIFS_inter_CAD_sym":0,
                "BO_inter_CAD_sym":0,
                "CH_DIFS":False,
                "active_BO_when_busy":False,
                "passive_BO_when_busy":False,
                "BO_when_clear":True,
                "active_clear_BO":True,
                "CA_Change_Channel":True,
            },            
            

            "xCANL_CAD":{


                "protocol":"CANL22",
                "n_retry":6,
                "abort_after_retries":False,
                "Wcanl_min_sym":0,
                "Wcanl_max_sym":50,

                "nCAD_DIFS":2,      
                "DIFS_inter_CAD_sym":5,    
                "CH_DIFS":True,

                "CANL22_P": 0,
                "CANL22_L1_min": 6,
                "CANL22_L1_MAX": 16,
                "CANL22_L2": 6,
                "CANL22_check_busy": True,
                "CANL22_RTS_min_payload_size": 100000000000,
                "CANL22_fair_factor": 1, # 0 means no reduction
                "CANL22_softer_fair": False,

                "CANL22_RTS_hdr_size": 4, 
                "CANL22_RTS_PHY_HDR": 1,
                "CANL22_data_hdr_size": 0,

                "CANL22_MAX_NAV_x_chan_threshold": 2800, # ms         

                "Wclear_min_sym":0,
                "Wclear_max_sym":20,
                "BO_when_clear":True,
                "active_clear_BO":False,


                "Interrupts_on_header_valid":True,

            },
            "ideal_FIFO":{
                "protocol":"ideal_FIFO",
            },
            "ALOHA":{
                "protocol":"ALOHA",
            },
            "CANL_2022":{
                "protocol":"CANL22",
                "n_retry":5,
                "abort_after_retries":True,#True,
                # "Wcanl_min_sym":0,
                # "Wcanl_max_sym":6,
                "nCAD_DIFS":0,      
                "DIFS_inter_CAD_sym":0,    

                "CANL22_P": 0,
                "CANL22_L1_min": 5,
                "CANL22_L1_MAX": 120,
                # "CANL22_L2": 6,
                "CANL22_check_busy": False,
                "CANL22_RTS_min_payload_size": 120000000,
                # "CANL22_fair_factor": 4, # 0 means no reduction
                "CANL22_fair_factor": 1, # 0 means no reduction
                "CANL22_softer_fair": False,

                "CANL22_RTS_hdr_size": 4, 
                "CANL22_RTS_PHY_HDR": False, #(implicit header for rts (header is in upper layer))
                "CANL22_data_hdr_size": 4,

                "CANL22_MAX_NAV_x_chan_threshold": 22222200, # ms       

                "BO_when_clear":False,  

                "Interrupts_on_header_valid":True,
            },

        }

    }


    res3=main_with_params(params)

    sys.stdout=stdout_print_target

    for key in res3["TOTAL"]:
        print("{0}:".format(key))
        if isinstance(res3["TOTAL"][key], list):
            if len(res3["TOTAL"][key])>12:
                print(res3["TOTAL"][key][:12])
            else:
                print(res3["TOTAL"][key])
        else:
            print(res3["TOTAL"][key])

    print("")
    print("")
    print("")
    print(start_time)


    pickle.dump(params, open('results/{0}_params.dat'.format(start_time), 'wb'))
    pickle.dump(res3, open('results/{0}_data.dat'.format(start_time), 'wb'))

    if JSON_EXPORT:
        json_dict = json.dumps(res3, indent=4)
        with open('results/{0}_results.json'.format(start_time), "w") as outfile:
            outfile.write(json_dict)

