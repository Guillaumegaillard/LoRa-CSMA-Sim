# -*- coding: utf-8 -*-
######################### Module to build topologies for the LoRa-CSMA-Sim Simulator #######################################
######################### https://github.com/Guillaumegaillard/LoRa-CSMA-Sim ###############################################
######################### 2025-04-01 #######################################################################################
######################### License MIT: #####################################################################################
######################### https://github.com/Guillaumegaillard/LoRa-CSMA-Sim/blob/main/LICENSE #############################

import numpy as np
import math
import constants
import sys

rng = np.random.default_rng()

# A switch for further computations
lora24GHz = False


# build a crown of equi-distant nodes
def get_equi_distrib(nb_nodes,center_x,center_y,radius):
    nodes=[]
    
    alpha=2*math.pi/nb_nodes
    for i in range(0,nb_nodes):
        current_angle=i*alpha
        posx = radius*math.cos(current_angle)+center_x
        posy = radius*math.sin(current_angle)+center_y

        nodes.append({"id":i,"x":posx,"y":posy})                

    return(nodes)


# spread uniformely on a disk, (avoiding obstacle disks), avoiding overlaps (min inter distance)
def get_non_overlap_unif_distrib(nb_nodes,center_x,center_y,maxDist,minInterDist=-1,obstacle_disks=[]):

    if minInterDist == -1:#default
        # devices get a minimum inter distance (no overlap)
        min_inter_dist=10/nb_nodes*20
    else:
        min_inter_dist=minInterDist


    nodes=[]
    maxrounds=200
    for i in range(0,nb_nodes):
        # this is a prodecure for placing nodes
        # and ensure minimum distance between each pair of nodes
        found = 0
        rounds = 0
        while (found == 0):# and rounds < 100):
            a = .99*rng.random()+0.01 #avoid log10(0)/dividebyzero
            b = .99*rng.random()+0.01 #avoid log10(0)/dividebyzero
            if b<a:
                a,b = b,a
            posx = b*maxDist*math.cos(2*math.pi*a/b)+center_x
            posy = b*maxDist*math.sin(2*math.pi*a/b)+center_y
            if len(nodes) > 0:
                broken=False
                for obs_x,obs_y,obs_mr in obstacle_disks:
                    dist = np.sqrt((obs_x-posx)**2+(obs_y-posy)**2)
                    if dist < obs_mr:
                        rounds = rounds + 1
                        if rounds == maxrounds:
                            print("could not place new node, giving up")
                            exit(-1)
                        broken=True
                        break 
                if not broken:
                    for index, n in enumerate(nodes):
                        dist = np.sqrt(((abs(n["x"]-posx))**2)+((abs(n["y"]-posy))**2))
                        if dist < min_inter_dist:
                            rounds = rounds + 1
                            if rounds == maxrounds:
                                print("could not place new node, giving up")
                                exit(-1)
                            broken=True
                            break
                # if index==len(nodes)-1:
                if not broken:
                    found = 1
                    x = posx
                    y = posy
            else:
                # print("first node")
                x = posx
                y = posy
                found = 1
        nodes.append({"id":i,"x":x,"y":y})                

    return(nodes)


# build a uniformly spread 2D disk topo based on: 
## a number of devices *nb_nodes*
## a number of GWs *nb_gws* (distributed on a circle of radius (maxDist*(1-1/(nb_gws-1)))) and on center (if >1 else just on center)
## a PHY layer setting *experiment* (inherited from LoRaSim)
## an optional maximum distance to the center, *static_maxDist*
#### default maxDist is max. range of a noiseless GW-ED link for the given experiment
### cluster_params= (clustering algorithm, number of clusters, distance, (min, max) nodes per cluster
def build_topo(nb_nodes,nb_gws,experiment,static_maxDist=0,cluster_params=('none',0,0,(0,0))):
    topo={}

    # if no max has been given
    if static_maxDist==0:
        if lora24GHz:
            Ptx=constants.Ptx_2dot4GHz
        else:
            Ptx=constants.Ptx_subGHz

        if lora24GHz:
            if experiment in [0,1,4,6,7]:
                minsensi = constants.sensi_2dot4GHz[7,2]     # 7th row is SF12, 2nd column is BW203
            elif experiment == 2:
                minsensi = constants.sensi_2dot4GHz[0,5]     # row 0 is SF5, 5th column is BW1625
            elif experiment in [3,5]:
                minsensi = np.amin(constants.sensi_2dot4GHz) ## Experiment 3 can use any setting, so take minimum
        else:
            if experiment in [0,1,4,6,7]:
                minsensi = constants.sensi_subGHz[6,2]     # 6th row is SF12, 2nd column is BW125
            elif experiment == 2:
                minsensi = constants.sensi_subGHz[0,4]     # first row is SF6, 4th column is BW500
            elif experiment =="SF7BW500CAD4":
                minsensi = constants.sensi_subGHz[1,3]     # second row is SF7, 4th column is BW500
            elif experiment in [3,5]:
                minsensi = np.amin(constants.sensi_subGHz) ## Experiment 3 can use any setting, so take minimum
            
        Lpl = Ptx - minsensi
        print("amin", minsensi, "Lpl", Lpl)
        maxDist = constants.d0*(math.e**((Lpl-constants.Lpld0)/(10.0*constants.gamma_GW)))
        
        topo["amin"]=minsensi
        topo["Lpl"]=Lpl
        topo["maxDist"]=maxDist
    else:
        maxDist = static_maxDist 
        topo["maxDist"]=maxDist

    # topo center coords
    center_x = maxDist+10
    center_y = maxDist+10

    topo["center"]={
        "center_x":center_x,
        "center_y":center_y,
    }

    # base station(s) (GW(s)) placement
    ## all GWs form a crown (but one, placed at center, when there are 1 or more than 3 GWs) 
    if nb_gws == 1:
        # topo['GWs'].append({"id":0,"x":center_x,"y":center_y})
        topo['GWs'] = [{"id":0,"x":center_x,"y":center_y}]
    elif nb_gws == 2 or nb_gws == 3:
        topo['GWs'] = get_equi_distrib(nb_gws,center_x,center_y,maxDist-maxDist/(nb_gws))
    else:
        topo['GWs'] = get_equi_distrib(nb_gws-1,center_x,center_y,maxDist-maxDist/(nb_gws-1))
        topo['GWs'].append({"id":nb_gws-1,"x":center_x,"y":center_y})


    #### default clustering is no cluster (id = -1)
    cluster_ids=[-1]*nb_nodes


    # clustering algorithms based on a pre-spread uniform 
    if cluster_params[0] in ['none','kmeans','cycle-shrink']: 
        nodes = get_non_overlap_unif_distrib(nb_nodes,center_x,center_y,maxDist) 

        # algo not in current use, requires a library
        if cluster_params[0] == 'kmeans' and cluster_params[1]!=0: # clusters
            from k_means_constrained import KMeansConstrained as KMeans
            # k-means-constrained : https://joshlk.github.io/k-means-constrained/
            # K-means clustering implementation whereby a minimum and/or maximum size for each cluster can be specified.
            # This K-means implementation modifies the cluster assignment step (E in EM) by formulating it as a Minimum Cost Flow (MCF) linear network optimisation problem. This is then solved using a cost-scaling push-relabel algorithm and uses Google's Operations Research tools's SimpleMinCostFlow which is a fast C++ implementation.
            # This package is inspired by Bradley et al.. The original Minimum Cost Flow (MCF) network proposed by Bradley et al. has been modified so maximum cluster sizes can also be specified along with minimum cluster size.
            # The code is based on scikit-lean's KMeans and implements the same API with modifications.
            # Ref:
            #     Bradley, P. S., K. P. Bennett, and Ayhan Demiriz. "Constrained k-means clustering." Microsoft Research, Redmond (2000): 1-8.
            #     Google's SimpleMinCostFlow C++ implementation


            # from sklearn.cluster import KMeans


            coords=np.zeros((nb_nodes,2))

            for node in nodes:
                coords[node["id"]][0]=node["x"]
                coords[node["id"]][1]=node["y"]

            # print(coords)
            
            # define the model
            kmeans_model = KMeans(
                n_clusters=cluster_params[1],
                size_min=cluster_params[3][0],
                # size_max=cluster_params[3][1]
                )


            # assign each data point to a cluster
            kmeans_result = kmeans_model.fit_predict(coords)


            ccs_dict={}
            for cluster in range(cluster_params[1]):
                # print(np.count_nonzero((kmeans_result-cluster)==0))
                ccs_dict[cluster]={"dists":[],"neighs":[]}
                # print(cluster, kmeans_model.cluster_centers_[cluster][0])

            for node in nodes:
                cluster=kmeans_result[node["id"]]

                cc_x=kmeans_model.cluster_centers_[cluster][0]
                cc_y=kmeans_model.cluster_centers_[cluster][1]

                ccs_dict[cluster]["neighs"].append(node["id"])
                ccs_dict[cluster]["dists"].append( ((cc_x-node["x"])**2+(cc_y-node["y"])**2)**(1/2) )


            for cluster in range(cluster_params[1]):

                inds=np.argsort(ccs_dict[cluster]["dists"])[::-1]

                if len(inds)>cluster_params[3][1]:
                    to_rem=np.array(ccs_dict[cluster]["neighs"])[inds][:len(inds)-cluster_params[3][1]]
                    kmeans_result[to_rem]=-1


            cluster_ids=kmeans_result

        
        # form clusters by selecting them in a zone and placing them closer to each other
        elif cluster_params[0] in ['cycle-shrink']: 
            clusts={"centers":[],"populations":[],"dists":[]}


            # get cluster sizes 
            degrees= rng.integers(cluster_params[3][0],cluster_params[3][1],size=cluster_params[1])
            # add them equal share of border nodes (remaining nodes)
            remaining=0
            if degrees.sum()<nb_nodes:
                remaining=nb_nodes-degrees.sum()
            equal_share = remaining // cluster_params[1]

            # build the clusters cyclicly 
            #       # first node uniformly in untaken nodes 
            clusts["centers"]=[(nodes[i]["x"],nodes[i]["y"]) for i in range(cluster_params[1])]
            # clusts["populations"]=[1]*cluster_params[1]
            for i in range(cluster_params[1]):
                cluster_ids[i]=i
                clusts["populations"].append([i])
                clusts["dists"].append([])

            #       # append in turn the untaken node closest to the center of cluster (and update center)
            current_clust=0
            for turn in range(cluster_params[1],degrees.sum()+equal_share*cluster_params[1]):
                if len(clusts["populations"][current_clust]) == degrees[current_clust]+equal_share:
                    current_clust += 1
                    current_clust %= cluster_params[1]

                closest=-1
                closest_dist=1000000000000
                for node in nodes:
                    if cluster_ids[node["id"]]==-1:
                        dist_to_center = ((clusts["centers"][current_clust][0]-node["x"])**2+(clusts["centers"][current_clust][1]-node["y"])**2)**(1/2)
                        if dist_to_center < closest_dist:
                            closest=node["id"]
                            closest_dist=dist_to_center 
                
                cluster_ids[closest]=current_clust
                clusts["populations"][current_clust].append(closest)
                clusts["centers"][current_clust] = (
                        np.mean([nodes[i]["x"] for i in clusts["populations"][current_clust]]),
                        np.mean([nodes[i]["y"] for i in clusts["populations"][current_clust]])
                    )

                current_clust += 1
                current_clust %= cluster_params[1]

            # print(clusts)
            # print(0/0)

            # remove border nodes

            for current_clust in range(cluster_params[1]):
                cc_x=clusts["centers"][current_clust][0]
                cc_y=clusts["centers"][current_clust][1]

                for node_id in clusts["populations"][current_clust]:
                    node=nodes[node_id]
                    clusts["dists"][current_clust].append( ((cc_x-node["x"])**2+(cc_y-node["y"])**2)**(1/2) )


            for current_clust in range(cluster_params[1]):

                inds=np.argsort(clusts["dists"][current_clust])[::-1]
                to_rem=np.array(clusts["populations"][current_clust])[inds][:equal_share]
                # to_keep = np.array(clusts["populations"][current_clust])[inds][equal_share:]
                clusts["populations"][current_clust] = np.array(clusts["populations"][current_clust])[inds][equal_share:]

                clusts["centers"][current_clust] = (
                        np.mean([nodes[i]["x"] for i in clusts["populations"][current_clust]]),
                        np.mean([nodes[i]["y"] for i in clusts["populations"][current_clust]])
                    )

                cc_x=clusts["centers"][current_clust][0]
                cc_y=clusts["centers"][current_clust][1]

                clusts["dists"][current_clust]=[]
                for node_id in clusts["populations"][current_clust]:
                    node=nodes[node_id]
                    clusts["dists"][current_clust].append( ((cc_x-node["x"])**2+(cc_y-node["y"])**2)**(1/2) )

                for node_id in to_rem:
                    cluster_ids[node_id]=-1

            # shrink around centers
                # compute shrink factor from max of max dist to centers
            shrink_factor = cluster_params[2] / max([max(clusts["dists"][current_clust]) for current_clust in range(cluster_params[1])])

            for current_clust in range(cluster_params[1]):
                for node_id in clusts["populations"][current_clust]:
                    if nodes[node_id]["x"]>clusts["centers"][current_clust][0]:
                        distx=nodes[node_id]["x"]-clusts["centers"][current_clust][0]
                        nodes[node_id]["x"] -= distx*shrink_factor
                    else:
                        distx=clusts["centers"][current_clust][0] - nodes[node_id]["x"]
                        nodes[node_id]["x"] += distx*shrink_factor

                    if nodes[node_id]["y"]>clusts["centers"][current_clust][1]:
                        disty=nodes[node_id]["y"]-clusts["centers"][current_clust][1]
                        nodes[node_id]["y"] -= disty*shrink_factor
                    else:
                        disty=clusts["centers"][current_clust][1] - nodes[node_id]["y"]
                        nodes[node_id]["y"] += disty*shrink_factor
                        


    # clustering algorithms based on a pre-spread uniform 
    elif cluster_params[0] in ['manual-uniform','manual-equi-uniform']: 
        if cluster_params[0] in ['manual-uniform']:
            centers = get_non_overlap_unif_distrib(cluster_params[1],center_x,center_y,maxDist-cluster_params[2],minInterDist=2*cluster_params[2])
        elif cluster_params[0] in ['manual-equi-uniform']: 
            centers = get_equi_distrib(cluster_params[1],center_x,center_y,maxDist-cluster_params[2])

        nodes=centers[:]

        cluster_ids=[i for i in range(cluster_params[1])]

        for clust_center_id in range(cluster_params[1]):
            clust_center=centers[clust_center_id]
            degree= rng.integers(cluster_params[3][0],cluster_params[3][1])
            nodes+=get_non_overlap_unif_distrib(degree,clust_center["x"],clust_center["y"],cluster_params[2]) 
            # nodes+=get_non_overlap_unif_distrib(degree,clust_center["x"],clust_center["y"],maxDist/cluster_params[1]) 
            cluster_ids+=[clust_center_id]*degree


        if len(nodes)<nb_nodes:
            remaining=nb_nodes-len(nodes)
            nodes += get_non_overlap_unif_distrib(remaining,center_x,center_y,maxDist, obstacle_disks = [(c["x"], c["y"], cluster_params[2]) for c in centers])
            cluster_ids += [-1]*remaining


        nodes=nodes[:nb_nodes]
        node_ids = np.arange(nb_nodes)
        rng.shuffle(node_ids)
        for node_id in range(nb_nodes):
            nodes[node_id]["id"]=node_ids[node_id]
            nodes[node_id]["cluster"]=cluster_ids[node_id]

        # print(len(cluster_ids),len(nodes))


    # store coordinates in the topology dict
    topo["nodes"]={}
    for node in nodes:
        topo["nodes"][node["id"]]={
            "x":node["x"],
            "y":node["y"], 
            "cluster":node["cluster"] if "cluster" in node else cluster_ids[node["id"]] # manual-uniform requires re-shuffling so adds "cluster" in node's dict
            # "cluster":cluster_ids[node["id"]]
        }

    return topo


if __name__ == '__main__':
    nb_nodes=20
    nb_gws=5
    experiment=4

    # topopo=build_topo(nb_nodes,experiment,static_maxDist=0)
    topopo=build_topo(nb_nodes,nb_gws,experiment,static_maxDist=100)
    print(topopo)
    # print(0/0)

