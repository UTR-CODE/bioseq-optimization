"""Promoter optimization benchmark for DREAM using 60 WeakSeqCNN models."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr
from torch import nn
from torch.nn import functional as F

from BioSeqBench.evaluation.metrics import compute_metrics
from BioSeqBench.models.weakseqcnn import WeakSeqCNN
from BioSeqBench.oracles.dream import DreamAutosomeOracle
from BioSeqBench.optimization.tfbind8_benchmark import (
    AutoregressiveDNA, DeepExplorationGenerator, _adalead_parent_mask,
    _fast_seqprop_logits, _fit_autoregressive, _straight_through_sample,
)
from BioSeqBench.training.train_yeast_dream2022_weak_cnns import _features_from_codes, _split_visible_pool

LENGTH = 110
DNA = np.asarray(list("ACGT"))
ALGORITHMS = ("ism_greedy", "simulated_annealing", "genetic_algorithm", "adalead", "vanilla_activation_maximization", "fast_seqprop", "cbas", "den")


def _seed(seed: int) -> np.random.Generator:
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


def _seqs(codes: np.ndarray) -> np.ndarray:
    return np.asarray(["".join(row) for row in DNA[np.asarray(codes, dtype=np.int64)]], dtype=str)


def _codes(sequences: Sequence[str]) -> np.ndarray:
    lookup = {b:i for i,b in enumerate("ACGT")}
    return np.asarray([[lookup[b] for b in str(s)] for s in sequences], dtype=np.uint8)


def _random(n: int, rng: np.random.Generator) -> np.ndarray:
    return _seqs(rng.integers(0, 4, size=(n, LENGTH), dtype=np.uint8))


def _mutate(sequences: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    c = _codes(sequences); p = rng.integers(0, LENGTH, len(c)); o = rng.integers(1, 4, len(c)); rows=np.arange(len(c)); c[rows,p]=(c[rows,p]+o)%4; return _seqs(c)


def _onehot(sequences: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    c = torch.as_tensor(_codes(sequences), device=device, dtype=torch.long)
    return F.one_hot(c, num_classes=4).float(), torch.ones((len(c), LENGTH), dtype=torch.bool, device=device)


class Tracker:
    def __init__(self, excluded: set[str], *, max_queries: int, minimum_queries: int, patience: int, target_fraction: float):
        self.excluded=excluded; self.max_queries=max_queries; self.minimum_queries=minimum_queries; self.patience=patience; self.target_fraction=target_fraction
        self.archive={}; self.rows=[]; self.snapshots=[]; self.last_improvement=0; self.best=-np.inf
    def observe(self, query: int, seq: np.ndarray, score: np.ndarray):
        for s,v in zip(seq,score):
            if str(s) not in self.excluded and (str(s) not in self.archive or float(v)>self.archive[str(s)]): self.archive[str(s)]=float(v)
        if len(self.archive)<100:return
        selected=sorted(self.archive.items(), key=lambda x:(-x[1],x[0]))[:100]; mean=float(np.mean([x[1] for x in selected]))
        if mean>self.best+1e-8:
            self.best=mean; self.last_improvement=query; self.rows.append({'query_count':query,'top100_mean_weak_prediction':mean}); self.snapshots.append(np.asarray([x[0] for x in selected],dtype=str))
    def done(self, query:int)->bool:
        return query>=self.max_queries or (self.rows and query>=self.minimum_queries and query-self.last_improvement>=self.patience)
    def finalize(self, algorithm:str, query:int, metadata:dict, *, selection_fraction:float=0.90):
        if not self.rows: raise RuntimeError(f'{algorithm} produced fewer than 100 novel sequences')
        y=np.asarray([r['top100_mean_weak_prediction'] for r in self.rows]); target=y[0]+selection_fraction*(y[-1]-y[0]); idx=int(np.flatnonzero(y>=target-1e-12)[0]); selected=self.snapshots[idx]
        return selected, int(query), pd.DataFrame(self.rows).assign(is_q90=lambda x:False), {'q90_query_count':int(self.rows[idx]['query_count']),'saturation_top100_mean':float(y[-1]),'q90_top100_mean':float(y[idx]),'selection_rule':f'earliest top-100 archive reaching {selection_fraction:.0%} saturation gain',**metadata}


class Surrogate:
    def __init__(self, model: WeakSeqCNN, mean: float, scale: float, device: torch.device, batch: int=4096): self.model=model; self.mean=mean; self.scale=scale; self.device=device; self.batch=batch; self.query_count=0
    @torch.no_grad()
    def query(self, seq: np.ndarray):
        values=np.asarray(seq,dtype=str); out=[]
        for start in range(0,len(values),self.batch):
            stop=min(start+self.batch,len(values)); features,mask=_onehot(values[start:stop],self.device); out.append((self.model(features,mask)*self.scale+self.mean).cpu().numpy())
        self.query_count+=len(values); return np.concatenate(out)


def _basic(model, initial, excluded, algorithm, seed, device, max_queries, min_queries, patience, target_fraction, *, fixed_budget=False):
    rng=_seed(seed); state=torch.load(model[0],map_location='cpu',weights_only=False); net=WeakSeqCNN(**state['model_config']).to(device).eval(); net.load_state_dict(state['model_state']); surrogate=Surrogate(net,float(state['train_label_mean']),float(state['train_label_std']),device); tracker=Tracker(excluded,max_queries=max_queries,minimum_queries=min_queries,patience=patience,target_fraction=target_fraction)
    current=initial.copy(); scores=surrogate.query(current); tracker.observe(surrogate.query_count,current,scores)
    if algorithm=='ism_greedy':
        mutant_batch_cost = LENGTH * 3 * len(current)
        while surrogate.query_count + mutant_batch_cost <= max_queries:
            blocks=[]
            for s in current:
                c=_codes([s])[0]; m=np.repeat(c[None,:], LENGTH*3, axis=0); k=0
                for p in range(LENGTH):
                    for b in range(4):
                        if b!=c[p]: m[k,p]=b; k+=1
                blocks.append(m)
            all_mutants=_seqs(np.concatenate(blocks)); values=surrogate.query(all_mutants)
            tracker.observe(surrogate.query_count,all_mutants,values)
            offset=0
            for i,block_codes in enumerate(blocks):
                block_values=values[offset:offset+LENGTH*3]; j=int(np.argmax(block_values))
                if block_values[j] > scores[i]:
                    current[i]=_seqs(block_codes[j:j+1])[0]; scores[i]=block_values[j]
                offset += LENGTH*3
            if tracker.done(surrogate.query_count): break
    elif algorithm in {'vanilla_activation_maximization','fast_seqprop'}:
        logits=nn.Parameter(_onehot(initial,device)[0]+0.05*torch.randn(100,LENGTH,4,device=device)); params=[logits]
        if algorithm=='fast_seqprop':
            gamma=nn.Parameter(torch.ones(100,4,device=device)); beta=nn.Parameter(torch.zeros(100,4,device=device)); params += [gamma,beta]
        opt=torch.optim.Adam(params,lr=0.1); step=0
        while surrogate.query_count+100<=max_queries:
            opt.zero_grad(set_to_none=True); scaled=logits
            if algorithm=='fast_seqprop': scaled=logits*gamma[:,None,:]+beta[:,None,:]
            probs=torch.softmax(scaled,dim=-1); st,hard=_straight_through_sample(probs)
            pred=net(st)*surrogate.scale+surrogate.mean; (-pred.mean()).backward(); opt.step(); step+=1
            sequences=_seqs(hard.detach().argmax(-1).cpu().numpy()); vals=pred.detach().cpu().numpy(); surrogate.query_count+=100; tracker.observe(surrogate.query_count,sequences,vals)
            if tracker.done(surrogate.query_count): break
    elif algorithm=='cbas':
        encoded=torch.as_tensor(_codes(initial),device=device,dtype=torch.long); prior=AutoregressiveDNA(LENGTH).to(device); _fit_autoregressive(prior,encoded,epochs=10,learning_rate=3e-3); search=prior; rounds=0
        while surrogate.query_count+1000<=max_queries:
            samples=search.sample(1000,device); sequences=_seqs(samples.cpu().numpy()); vals=surrogate.query(sequences); tracker.observe(surrogate.query_count,sequences,vals); threshold=float(np.quantile(vals,0.8)); weights=torch.ones(1000,device=device); _fit_autoregressive(search,samples,epochs=5,learning_rate=1e-3,weights=weights,batch_size=1000); rounds+=1
            if tracker.done(surrogate.query_count): break
    elif algorithm=='den':
        gen=DeepExplorationGenerator(16,LENGTH).to(device); noise=torch.randn(100,16,device=device); target=torch.as_tensor(_codes(initial),device=device,dtype=torch.long); pre=torch.optim.Adam(gen.parameters(),lr=3e-3)
        for _ in range(30):
            loss=F.cross_entropy(gen(noise).reshape(-1,4),target.reshape(-1)); pre.zero_grad(set_to_none=True); loss.backward(); pre.step()
        opt=torch.optim.Adam(gen.parameters(),lr=3e-3)
        while surrogate.query_count+100<=max_queries:
            probs=torch.softmax(gen(noise),dim=-1); st,hard=_straight_through_sample(probs); pred=net(st)*surrogate.scale+surrogate.mean; loss=-pred.mean(); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); surrogate.query_count+=100; sequences=_seqs(hard.detach().argmax(-1).cpu().numpy()); tracker.observe(surrogate.query_count,sequences,pred.detach().cpu().numpy());
            if tracker.done(surrogate.query_count): break
    else:
        while surrogate.query_count+100<=max_queries:
            if algorithm=='simulated_annealing': proposals=_mutate(current,rng)
            elif algorithm=='genetic_algorithm':
                elite=current[np.argsort(scores)[-20:]]; a=_codes(elite[rng.integers(0,20,100)]); b=_codes(elite[rng.integers(0,20,100)]); c=np.where(rng.random(a.shape)<.5,a,b); mask=rng.random(c.shape)<.02; c=np.where(mask,(c+rng.integers(1,4,c.shape))%4,c); proposals=_seqs(c)
            elif algorithm=='adalead': proposals=_mutate(current,rng)
            else: proposals=_mutate(current,rng)
            vals=surrogate.query(proposals); tracker.observe(surrogate.query_count,proposals,vals)
            if algorithm=='simulated_annealing':
                temp=max(np.std(scores)*(.001**(surrogate.query_count/max_queries)),1e-6); accept=(vals>=scores)|(rng.random(100)<np.exp(np.minimum((vals-scores)/temp,0))); current[accept]=proposals[accept]; scores[accept]=vals[accept]
            else:
                keep=np.argsort(np.concatenate([scores,vals]))[-100:]; current=np.concatenate([current,proposals])[keep]; scores=np.concatenate([scores,vals])[keep]
            if tracker.done(surrogate.query_count):break
    if fixed_budget and surrogate.query_count < max_queries:
        # Complete the common budget with uniformly random novel proposals when
        # an algorithm's native batch granularity (e.g. DREAM ISM's 330-mutant
        # sweep) cannot land exactly on the budget boundary.
        while surrogate.query_count < max_queries:
            count = min(100, max_queries - surrogate.query_count)
            fill = _random(count, rng)
            values = surrogate.query(fill)
            tracker.observe(surrogate.query_count, fill, values)
    seq,q,tr,meta=tracker.finalize(algorithm,surrogate.query_count,{'runtime_seconds':0.0},selection_fraction=1.0 if fixed_budget else target_fraction); return seq, np.full(len(seq), np.nan), q, tr, meta


def run_dream_optimization(*, checkpoint_root='BioSeqBench/results/dream2022_weak_cnns', data_root='BioSeqBench/dataset/yeast_gpra_dream2022', strong_checkpoint='BioSeqBench/checkpoints/strong_yeast_dream2022_autosome_v3_validation_5k/yeast_gpra_dream2022/autosome_legnet/best_val.pt', output_dir='BioSeqBench/results/dream2022_optimization_benchmark', fractions=(.01,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95), max_queries=200000, minimum_queries=20000, plateau_patience=50000, target_fraction=.90, num_replicates=5, device='cuda', resume=True, fixed_budget=False):
    device=torch.device(device); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); train=pd.read_csv(Path(data_root)/'train.csv',usecols=['sequence','label','length']); seq=train.sequence.astype(str).to_numpy(); labels=train.label.to_numpy(float); lengths=train.length.to_numpy(int); order=np.lexsort((np.arange(len(train)),labels)); oracle=DreamAutosomeOracle(strong_checkpoint,device=device); protocol={'dataset':'yeast_gpra_dream2022','algorithms':list(ALGORITHMS),'num_replicates':num_replicates,'num_candidates':100,'sequence_length':LENGTH,'saturation_fraction':1.0 if fixed_budget else target_fraction,'fixed_budget':fixed_budget,'max_queries':max_queries,'plateau_patience_queries':plateau_patience,'initialization':'50 top-label sequences from actual fit split (restricted to length 110) + 50 full-space random 110-mers','weak_model_policy':'each checkpoint member independently','strong_model':'Autosome LegNet best_val.pt'}; (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n'); summaries=[]
    for fi,f in enumerate(fractions):
        nvis=round(len(train)*f); visible_indices=order[:nvis]; fit_local,_=_split_visible_pool(nvis,.1,20260827); fit_indices=visible_indices[fit_local]; fit_indices=fit_indices[lengths[fit_indices]==LENGTH]; top=seq[fit_indices[np.argsort(labels[fit_indices])[-50:]][::-1]]; excluded=set(seq[visible_indices].tolist());
        for rep in range(num_replicates):
            run=out/f'bottom_{f*100:g}pct'/f'run_{rep:02d}'; run.mkdir(parents=True,exist_ok=True); init_seed=20260827+fi*100000+rep*1000+997; rng=np.random.default_rng(init_seed); randoms=[]; seen=set(top.tolist());
            while len(randoms)<50:
                for s in _random(50,rng):
                    if s not in seen: seen.add(s); randoms.append(s)
                    if len(randoms)==50:break
            initial=np.concatenate([top,np.asarray(randoms)]); pd.DataFrame({'sequence':initial,'source':['train_top']*50+['full_space_random']*50,'in_weak_train':np.isin(initial,list(excluded))}).to_csv(run/'shared_initial_sequences.csv',index=False)
            for ai,alg in enumerate(ALGORITHMS):
                path=run/f'{alg}_candidates.csv'; sp=run/f'{alg}_summary.json'
                if resume and path.exists() and sp.exists(): summaries.append(json.loads(sp.read_text())); continue
                model_paths=sorted((Path(checkpoint_root)/f'bottom_{f*100:g}pct').glob('member_*.pt')); model=[model_paths[rep]]; excluded_fit=set(seq[fit_indices].tolist()); start=time.perf_counter()
                if alg in {'ism_greedy','simulated_annealing','genetic_algorithm','adalead'}: cand,_,q,tr,meta=_basic(model,initial,excluded_fit,alg,20260827+fi*100000+rep*1000+ai*101,device,max_queries,minimum_queries,plateau_patience,target_fraction,fixed_budget=fixed_budget)
                else:
                    cand,_,q,tr,meta=_basic(model,initial,excluded_fit,alg,20260827+fi*100000+rep*1000+ai*101,device,max_queries,minimum_queries,plateau_patience,target_fraction,fixed_budget=fixed_budget)
                strong=oracle.predict(cand); summary={'dataset':'yeast_gpra_dream2022','weak_model_fraction':f,'replicate':rep,'weak_model_member':rep,'algorithm':alg,'query_count':q,'q90_query_count':meta['q90_query_count'],'mean_strong_score':float(strong.mean()),'median_strong_score':float(np.median(strong)),'best_strong_score':float(strong.max()),'top_1pct_count':int((strong>=np.quantile(strong,.99)).sum()),'runtime_seconds':time.perf_counter()-start,'weak_checkpoint':str(model[0]),**meta}; pd.DataFrame({'sequence':cand,'strong_score':strong}).to_csv(path,index=False); sp.write_text(json.dumps(summary,indent=2)+'\n'); summaries.append(summary); print(f'{f:.0%} rep={rep} {alg} q90={meta["q90_query_count"]:,} strong_mean={summary["mean_strong_score"]:.3f}',flush=True)
    frame=pd.DataFrame(summaries); frame.to_csv(out/'summary.csv',index=False); frame.groupby(['weak_model_fraction','algorithm']).mean(numeric_only=True).reset_index().to_csv(out/'aggregate_summary.csv',index=False); return frame
