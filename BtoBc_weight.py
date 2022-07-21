import numpy as np
import pandas as pd
#import root_pandas
#from root_pandas import read_root, to_root
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import math
import scipy
from scipy.optimize import curve_fit
from scipy import interpolate
import pickle
from six.moves import cPickle
import collections
import uproot

rc('font',**{'family':'serif','serif':['Roman']})
rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 18})

def MakeHistogram_1D(sample, bins, weights = None, normed = False, density = None, range = None) :
  hist = np.histogram(sample, bins = bins, normed = normed, weights = weights, density = density, range = range)
  return hist[0]

n_bins = 50

tree = 'DecayTree'

vars_list = ['B_ptasy_1.50','log(B_LoKi_FDCHI2_BPV)','log(D0_LoKi_FDCHI2_BPV)','Bach_P','Bach_PT','D0_PT','log(Bach_IPCHI2_OWNPV)','log10(B_LoKi_MIPCHI2_PV)','log(D0_LoKi_MIPCHI2_PV)','B_LoKi_MAXDOCA','D0_LoKi_AMAXDOCA','log(1-B_LoKi_DIRA_BPV)','log(1-D0_LoKi_DIRA_BPV)','log10(B_LoKi_IP_BPV)','log10(D0_LoKi_IP_BPV)','log10(B_LoKi_RHO_BPV)','log10(B_LoKi_FD_BPV)','log10(D0_LoKi_RHO_BPV)','BDT','B_D0constPVconst_M','B_LoKi_LT_BPV']

vars_list_ratio = ['B_LoKi_LT_BPV']

var_MC = vars_list.copy()
var_MC.append('Bach_PIDK_corr')

var_data = vars_list.copy()
var_data.append('Bach_PIDK')

ranges = {'B_LoKi_LT_BPV':[0,0.007]}
label = {'B_LoKi_LT_BPV' : "$B_{LT}$"}

#df_Bmc = read_root('',tree)
#df_Bcmc = read_root('',tree)

Bmc_file = uproot.open('/data/lhcb/users/abrantes/Bc2DX_crosscheck/BDT_efficiency/Reweight/full_dataset/tuples/MC/Total_Bu2DPi_Cuts_PIDCorr_xgboostBDT_masscuts_fulldataset.root', branches=vars_list)
Bcmc_file = uproot.open('/data/lhcb/users/abrantes/Bc2DX_crosscheck/BDT_efficiency/Reweight/full_dataset/tuples/MC/Total_Bc2DPi_Cuts_PIDCorr_xgboostBDT_masscuts_fulldataset.root', branches=vars_list)

Bmc_tree = Bmc_file['DecayTree']
Bcmc_tree = Bcmc_file['DecayTree']


df_Bmc = Bmc_tree.arrays(vars_list, library="pd")
df_Bcmc = Bcmc_tree.arrays(vars_list, library="pd")

# apply cuts
Bmass_window = (5150,5380)
Bcmass_window = (6150,6380)

df_Bmc = df_Bmc.loc[(df_Bmc['B_D0constPVconst_M'] >= Bmass_window[0]) & (df_Bmc['B_D0constPVconst_M'] <= Bmass_window[1])]
df_Bcmc = df_Bcmc.loc[(df_Bcmc['B_D0constPVconst_M'] >= Bcmass_window[0]) & (df_Bcmc['B_D0constPVconst_M'] <= Bcmass_window[1])]

df_Bmc.dropna(inplace=True)
df_Bcmc.dropna(inplace=True)

#label = {'B_ptasy_1.50' : 'B_ptasy_1.50',
#        'log(B_LoKi_FDCHI2_BPV)' : 'log(B_LoKi_FDCHI2_BPV)',
#        'log(D0_LoKi_FDCHI2_BPV)' : 'log(D0_LoKi_FDCHI2_BPV)',
#        'Bach_P' : 'Bach_P',
#        'Bach_PT' : 'Bach_PT',
#        'D0_PT' : 'D0_PT',
#        'log(Bach_IPCHI2_OWNPV)' : 'log(Bach_IPCHI2_OWNPV)',
#        'log10(B_LoKi_MIPCHI2_PV)' : 'log10(B_LoKi_MIPCHI2_PV)',
#        'log(D0_LoKi_MIPCHI2_PV)' : 'log(D0_LoKi_MIPCHI2_PV)',
#        'B_LoKi_MAXDOCA' : 'B_LoKi_MAXDOCA',
#        'D0_LoKi_AMAXDOCA' : 'D0_LoKi_AMAXDOCA',
#        'log(1-B_LoKi_DIRA_BPV)' : 'log(1-B_LoKi_DIRA_BPV)',
#        'log(1-D0_LoKi_DIRA_BPV)' : 'log(1-D0_LoKi_DIRA_BPV)',
#        'log10(B_LoKi_IP_BPV)' : 'log10(B_LoKi_IP_BPV)',
#        'log10(D0_LoKi_IP_BPV)' : 'log10(D0_LoKi_IP_BPV)',
#        'log10(B_LoKi_RHO_BPV)' : 'log10(B_LoKi_RHO_BPV)',
#        'log10(B_LoKi_FD_BPV)' : 'log10(B_LoKi_FD_BPV)',
#        'log10(D0_LoKi_RHO_BPV)' : 'log10(D0_LoKi_RHO_BPV)',
#        'BDT' : 'BDT',
#        'B_D0constPVconst_M' : 'B_D0constPVconst_M',
#        'B_LoKi_LT_BPV' : 'B_LoKi_LT_BPV',
#        'Bach_PIDK_corr' : 'Bach_PIDK_corr'}

label = {'B_LoKi_LT_BPV' : 'B LT'}

print('starting plots')

for v in vars_list_ratio:

    df_Bmc = df_Bmc[df_Bmc[v].between(df_Bmc[v].min(),df_Bmc[v].max())]

    fig, ax = plt.subplots(figsize=(7,7))

    #plt.errorbar(bin_centres, counts, yerr=err, fmt='o')
    plt.hist(df_Bcmc[v],bins=n_bins,histtype='step',color='green',label='Bc MC',range=ranges[v],density=True)
    plt.hist(df_Bmc[v],bins=n_bins,histtype='step',color='crimson',label='B MC',range=ranges[v],density=True)

    #plt.show()
    plt.xlabel(label[v])
    plt.xlim(ranges[v])
    plt.legend(fontsize=14)
    #plt.show()
    plt.savefig('compare_plots/%s.pdf' % v)
    plt.close()

df_Bmc['MC_weights'] = 1

tck_dict = {}

# ratio histograms

ratio_bins = 10

for v in vars_list_ratio:

    print("Starting variable: %s" %v)

    df_Bmc = df_Bmc[df_Bmc[v].between(ranges[v][0],ranges[v][1])]
    df_Bcmc = df_Bcmc[df_Bcmc[v].between(ranges[v][0],ranges[v][1])]

    Bmc_size = df_Bmc.count()[0]
    Bcmc_size = df_Bcmc.count()[0]

    # create ratio hists and apply weights
    binning = np.linspace(df_Bcmc[v].min(),df_Bcmc[v].max(),ratio_bins+1)

    Bcmc_vals, Bcmc_edges = np.histogram(df_Bcmc[v],bins=binning,density=True)
    Bmc_vals, Bmc_edges = np.histogram(df_Bmc[v],bins=binning,density=True,weights=df_Bmc['MC_weights'])

    ratio = Bcmc_vals/Bmc_vals

    Bmc_counts = Bmc_vals*Bmc_size
    Bcmc_counts = Bcmc_vals*Bcmc_size
    err = ratio * np.sqrt(1/abs(Bmc_counts) + 1/abs(Bcmc_counts))

    centres = (Bcmc_edges[:-1] + Bcmc_edges[1:])/2.

    # calculate spline
    tck = interpolate.splrep(centres, ratio, k=3, w=1.0/err)
    spline_x = np.linspace(df_Bmc[v].min(),df_Bmc[v].max(),1000)
    spline_val = interpolate.splev(spline_x, tck, der=0)

    fig, ax = plt.subplots(figsize=(7,7))
    plt.errorbar(centres,ratio,yerr=err,fmt='ko',markersize=3)
    plt.plot(spline_x,spline_val)
    plt.xlabel(label[v])
    plt.xlim(ranges[v])
    plt.savefig('ratios/%s_ratio.pdf' %v)
    plt.close()


    # calculate weight for each MC event and save
    Bmc_weights = interpolate.splev(df_Bmc[v].to_numpy(), tck)

    # Update weights
    df_Bmc['MC_weights'] = df_Bmc['MC_weights']*Bmc_weights

    print(df_Bmc['MC_weights'])
    #df_Bmc = df_Bmc.dropna()

    # Plot MCs with weigth
    plt.hist(df_Bcmc[v],bins=n_bins,histtype='step',color='green',label='Bc MC',range=ranges[v],density=True)
    plt.hist(df_Bmc[v],bins=n_bins,histtype='step',color='crimson',label='B MC - with weights',range=ranges[v],density=True,weights=df_Bmc['MC_weights'])

    #plt.show()
    plt.xlabel(label[v])
    plt.xlim(ranges[v])
    plt.legend(fontsize=14)
    #plt.show()
    plt.savefig('compare_plots/%s_weight.pdf' % v)
    plt.close()

    f = open('MC_with_weights.pkl', 'wb')
    cPickle.dump(df_Bmc, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

    #Store the polynomial order and the list of param values
    tck_dict[v] = tck
    f = open('splines/%s_tck_corrs.pkl' % v, 'wb')
    cPickle.dump(tck_dict[v], f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

# make histogram of final weights

weight_vals,weight_edges = np.histogram(df_Bmc['MC_weights'],bins=100,range=[0,1])
weight_centres = (weight_edges[:-1] + weight_edges[1:])/2.
Bmc_size = df_Bmc.count()[0]
#weight_err = np.sqrt(weight_vals/MC_size)
weight_err = np.sqrt(weight_vals)

fig, ax = plt.subplots(figsize=(7,7))
plt.errorbar(weight_centres,weight_vals,yerr=weight_err,fmt='ko',markersize=2,label='Weight')
plt.xlabel('Weight')
plt.ylabel('Arbitrary units')
plt.xlim(0,1)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('final_weights.pdf')
plt.close()

# save Bu MC with weights
mc_out = uproot.recreate("tuples/MC/Total_Bu2DPi_Cuts_PIDCorr_xgboostBDT_masscuts_fulldataset_LTreweighted.root")
mc_out["DecayTree"] = df_Bmc


# apply weight to Bu2DPi data
Bdata_file = uproot.open('/data/lhcb/users/abrantes/Bc2DX_crosscheck/BDT_efficiency/Reweight/full_dataset/tuples/data/Total_Bu2DPi_Cuts_xgboostBDT_masscuts_fulldataset.root', branches=var_data)

Bdata_tree = Bdata_file['DecayTree']

df_Bdata = Bdata_tree.arrays(var_data, library="pd")

df_Bdata['weights'] = 1
df_toweight = df_Bdata.copy()

df_Bdata['inrange'] = 1

for v in vars_list_ratio:
    print("Starting variable: %s" %v)

    df_toweight = df_toweight[(df_toweight[v] >= ranges[v][0]) & (df_toweight[v] <= ranges[v][1])]

    #check if event falls outside the range
    df_Bdata.loc[(df_Bdata[v]<ranges[v][0]) | (df_Bdata[v]>ranges[v][1]),"inrange"] = 0

    f = open('splines/%s_tck_corrs.pkl' % v, 'rb')
    tck = cPickle.load(f)
    f.close()

    df_toweight["tck_vals_{}".format(v)] = interpolate.splev(df_toweight[v], tck, der=0)
    df_toweight['weights'] = df_toweight['weights'] * df_toweight["tck_vals_{}".format(v)]
    print(df_toweight['weights'])

print('# Events before:',df_Bdata.count()[0])
print('# Events after:',df_toweight.count()[0])
print('Check ranges:',np.sum(df_Bdata['inrange']))
print('---------------------')
print('Max weight:',df_toweight['weights'].max())
print('Min weight:',df_toweight['weights'].min())

#append weights to dataframe, assign weight of 1 to events outside range
df_Bdata['weights'] = df_toweight['weights']
df_Bdata['weights'] = df_Bdata['weights'].fillna(1)

# Output dataframe with weights
#import awkward as ak
data_out = uproot.recreate("tuples/data/Total_Bu2DPi_Cuts_xgboostBDT_masscuts_fulldataset_LTreweighted.root")
data_out["DecayTree"] = df_Bdata

for v in vars_list_ratio:
    fig, ax = plt.subplots(figsize=(7,7))
    #plt.hist(df_Bcmc[v],bins=n_bins,histtype='step',color='red',label='Bc MC',range=ranges[v],density=True)
    plt.hist(df_Bdata[v],histtype='step',bins=n_bins,color='blue',label='B Data - no weights',range=ranges[v],density=True)
    plt.hist(df_Bdata[v],histtype='step',bins=n_bins,color='green',label='B Data - with weights',range=ranges[v],density=True,weights=df_Bdata['weights'])
    plt.xlabel(label[v])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), fancybox=False, shadow=False,ncol=2)
    plt.xlim(ranges[v])
    plt.ylim(bottom=0)
    plt.savefig('reweighted_plots/%s.pdf' %v)
    plt.close()

