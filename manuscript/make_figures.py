from pathlib import Path
import json, csv, statistics, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'manuscript'
FIG=OUT/'figures'; DATA=OUT/'data'
FIG.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)
analysis=json.load(open(ROOT/'artifacts/lifecycle-v25/analysis.json'))
records=analysis['records']
# flat record export
with open(DATA/'v25_records.csv','w',newline='') as f:
    fields=['id','dataset','condition','trigger','arm','seed','final_accuracy','lower_decile_client_utility','endpoint_asr','mean_after_stop_asr','mean_post_onset_asr','repair_admissions','repair_attempts','repair_seconds','transmitted_bytes','integrity_passed']
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows({k:r.get(k) for k in fields} for r in records)

def pct(x): return 100*x
# aggregate complete groups with bootstrap intervals already in analysis
rows=[]
for g in analysis['complete_seed_groups']:
    row={'dataset':g['dataset'],'condition':g['condition'],'trigger':g['trigger'],'arm':g['arm'],'seed_count':g['seed_count']}
    for k,v in g.items():
        if isinstance(v,dict) and 'mean' in v:
            row[k+'_mean']=v['mean']; row[k+'_lower']=v['lower']; row[k+'_upper']=v['upper']
    rows.append(row)
with open(DATA/'v25_group_summary.csv','w',newline='') as f:
    fields=sorted({k for r in rows for k in r}); w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


# Draw at the final printed width. Do not shrink a seven-inch chart into one column.
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'DejaVu Sans',
 'font.size':8,'axes.titlesize':8,'axes.labelsize':8,'xtick.labelsize':7.5,
 'ytick.labelsize':7.5,'legend.fontsize':7.5,'axes.edgecolor':'0.25',
 'axes.grid':True,'grid.color':'0.86','grid.linewidth':0.4})
arm_order=['quorum','clipping','periodic','finetune','fineprune','forced_periodic','forced_finetune','forced_fineprune']
labels={'quorum':'Quorum','clipping':'Clipping','periodic':'Periodic',
 'finetune':'Fine-tune','fineprune':'Fine-Pruning','forced_periodic':'Forced periodic',
 'forced_finetune':'Forced fine-tune','forced_fineprune':'Forced Fine-Pruning'}
markers=dict(zip(arm_order,['o','s','^','D','v','P','X','*']))
colors={a:str(.15+.65*i/(len(arm_order)-1)) for i,a in enumerate(arm_order)}
styles={'quorum':'--','periodic':'-','fineprune':':'}
linecolors={'quorum':'0.45','periodic':'0.05','fineprune':'0.25'}
datasets=['femnist','fashion_mnist']
dslabels={'femnist':'FEMNIST','fashion_mnist':'Fashion-MNIST'}

def save(fig,name):
 fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight',pad_inches=.03)
 fig.savefig(FIG/(name+'.png'),dpi=240,bbox_inches='tight',pad_inches=.03)
 plt.close(fig)

def group(ds,condition,trigger='patch',source=analysis):
 return {g['arm']:g for g in source['complete_seed_groups']
         if (g['dataset'],g['condition'],g['trigger'])==(ds,condition,trigger)}

fig,axs=plt.subplots(1,2,figsize=(3.3,2.9),sharex=True,sharey=True)
for ax,ds in zip(axs,datasets):
 g=group(ds,'attack_stop')
 for a in arm_order:
  ax.scatter(100*g[a]['mean_after_stop_asr']['mean'],100*g[a]['endpoint_asr']['mean'],
    s=27,marker=markers[a],color=colors[a],edgecolor='0.1',linewidth=.4,label=labels[a],zorder=3)
 ax.plot([0,100],[0,100],color='0.6',lw=.7,ls='--')
 ax.set(title=dslabels[ds],xlim=(-3,103),ylim=(-3,103),xticks=[0,50,100],yticks=[0,50,100])
axs[0].set_ylabel('Endpoint ASR (%)')
fig.supxlabel('Mean post-stop ASR (%)',y=.38,fontsize=8)
fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=2,
           frameon=False,handletextpad=.4,columnspacing=.7)
fig.subplots_adjust(left=.15,right=.98,top=.9,bottom=.53,wspace=.17)
save(fig,'endpoint_temporal_tradeoff')

# Only manifest-selected, completed histories are eligible; archived retries are excluded.
wanted=['quorum','periodic','fineprune'];hist={}
for r in records:
 if r['arm'] not in wanted or r['trigger']!='patch' or r['condition'] not in ['attack_stop','continuous']:continue
 folder=ROOT/'results/lifecycle-v25'/r['id']
 assert (folder/'completion.json').is_file()
 h=json.loads((folder/'summary.json').read_text())['targeted_attack_success_history']
 assert len(h)==100 and h[-1]==r['endpoint_asr']
 hist.setdefault((r['dataset'],r['condition'],r['arm']),[]).append(h)
for condition in ['attack_stop','continuous']:
 fig,axs=plt.subplots(2,1,figsize=(3.3,3.0),sharex=True,sharey=True)
 for ax,ds in zip(axs,datasets):
  for a in wanted:
   values=np.asarray(hist[(ds,condition,a)])*100;assert values.shape==(5,100)
   ax.plot(np.arange(1,101),values.mean(0),label=labels[a],color=linecolors[a],ls=styles[a],lw=1.15)
   ax.fill_between(np.arange(1,101),*np.percentile(values,[10,90],axis=0),color=linecolors[a],alpha=.10)
  for rr in [25,50,75,100]:ax.axvline(rr,color='0.65',lw=.6,ls=':')
  if condition=='attack_stop':ax.axvline(50,color='0.25',lw=.8,ls='--')
  ax.set(title=dslabels[ds],xlim=(1,100),ylim=(-2,103),yticks=[0,50,100])
  ax.set_ylabel('ASR (%)')
 axs[-1].set_xlabel('Round')
 fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=3,
            frameon=False,columnspacing=.7,handlelength=1.8)
 fig.suptitle('Attacker stops after round 50' if condition=='attack_stop' else 'Continuous poisoning',fontsize=8.5)
 fig.tight_layout(rect=(0,.08,1,.96),h_pad=.55)
 save(fig,'round_trajectories_'+condition)

fig,ax=plt.subplots(figsize=(3.3,2.55));x=np.arange(4);width=.24
combinations=[(ds,t) for ds in datasets for t in ['patch','checkerboard']]
for j,a in enumerate(wanted):
 vals=[100*group(ds,'continuous',t)[a]['mean_post_onset_asr']['mean'] for ds,t in combinations]
 ax.bar(x+(j-1)*width,vals,width,label=labels[a],color=['0.85','0.5','0.2'][j],
        edgecolor='0.15',linewidth=.4,hatch=['','//','..'][j])
ax.set_xticks(x,['FEMNIST\npatch','FEMNIST\nchecker','Fashion\npatch','Fashion\nchecker'])
ax.set_ylabel('Mean post-onset ASR (%)');ax.set_ylim(0,105)
fig.legend(*ax.get_legend_handles_labels(),loc='upper center',ncol=3,
           frameon=False,columnspacing=.7,handlelength=1.2)
fig.tight_layout(rect=(0,0,1,.88))
save(fig,'trigger_transfer_continuous')

# Auxiliary charts: all their values also occur in the main paper's policy table.
for name,metric in [('admission_rates','repair_admissions'),('utility_endpoint_attack_stop','final_accuracy')]:
 fig,axs=plt.subplots(2,1,figsize=(3.3,4.8),sharex=True)
 for ax,ds in zip(axs,datasets):
  g=group(ds,'attack_stop');scale=1 if metric=='repair_admissions' else 100
  ax.bar(np.arange(8),[scale*g[a][metric]['mean'] for a in arm_order],color=[colors[a] for a in arm_order],edgecolor='0.15',lw=.4)
  ax.set_title(dslabels[ds]);ax.set_ylabel('Admissions / 4' if scale==1 else 'Clean accuracy (%)')
  ax.set_ylim(0,4.3) if scale==1 else ax.set_ylim(65,90)
  if scale==100:
   twin=ax.twinx();twin.plot(np.arange(8),[100*g[a]['endpoint_asr']['mean'] for a in arm_order],color='0.1',marker='o',ms=3,lw=1)
   twin.set(ylim=(0,100),ylabel='Endpoint ASR (%)');twin.grid(False)
 axs[-1].set_xticks(np.arange(8),[labels[a] for a in arm_order],rotation=55,ha='right')
 fig.tight_layout();save(fig,name)

baseline=json.loads((ROOT/'artifacts/lifecycle-v26-coordinate/analysis.json').read_text())
fig,axs=plt.subplots(2,2,figsize=(3.3,3.5),sharex=True,sharey=True)
for i,ds in enumerate(datasets):
 for j,trig in enumerate(['patch','checkerboard']):
  ax=axs[i,j];g=group(ds,'continuous',trig,baseline)
  for k,a in enumerate(['quorum','periodic','nad','coordinate_median']):
   z=g[a];end=100*z['endpoint_asr']['mean'];temp=100*z['mean_post_onset_asr']['mean']
   ax.plot([k,k],[end,temp],color='0.55',lw=1.4)
   ax.errorbar(k,end,yerr=[[end-100*z['endpoint_asr']['lower']],[100*z['endpoint_asr']['upper']-end]],fmt='o',color='0.1',ms=3,capsize=2)
   ax.scatter(k,temp,facecolors='white',edgecolors='0.1',s=20,zorder=4)
  ax.set_title(dslabels[ds]+'\n'+trig)
  ax.set_xticks(range(4),['Q','P','NAD','Med'])
  ax.set_ylim(-3,103);ax.set_yticks([0,50,100])
  if j==0:ax.set_ylabel('ASR (%)')
fig.tight_layout(h_pad=.7,w_pad=.6);save(fig,'baseline_continuous')
print('Generated plots from',len(records),'main and',len(baseline['records']),'companion records')
