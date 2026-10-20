"""Digital64-trace STDP/LIF; two measured physical tiles feed one neuron step."""
import math
from experiments.mnist_conductance_20261019.core import GAINS


def address(neuron,feature):
    if type(neuron)is not int or type(feature)is not int or not 0<=neuron<4 or not 0<=feature<64:
        raise ValueError('Invalid logical synapse')
    return neuron+4*(feature//32),feature%32


def trajectory(schedule,voltages,gain):
    if len(schedule)!=32 or len(voltages)!=32 or gain not in GAINS:raise ValueError('Invalid LIF inputs')
    v=post_trace=0.;pre_trace=[0.]*64;eligibility=[0.]*64;steps=[];decay=math.exp(-.25)
    for tick,(mask,voltage) in enumerate(zip(schedule,voltages)):
        if not math.isfinite(voltage):raise ValueError('Nonfinite measured voltage')
        pre=[int(i in mask) for i in range(64)];before=v;u=.5*v+gain*voltage;post=int(u>=1)
        v=0. if post else u
        pre_trace=[a*decay for a in pre_trace];post_trace*=decay
        delta=[a*post-1.05*post_trace*s for a,s in zip(pre_trace,pre)]
        eligibility=[e+d for e,d in zip(eligibility,delta)]
        pre_trace=[a+s for a,s in zip(pre_trace,pre)];post_trace+=post
        steps.append(dict(t=tick,pre=pre,voltage_v=voltage,gain=gain,u_before=before,u_unreset=u,u_after=v,
            post=post,pre_trace=list(pre_trace),post_trace=post_trace,eligibility_increment=delta,eligibility=list(eligibility)))
    feature=min(range(64),key=lambda i:(-abs(eligibility[i]),i))
    choice=None if abs(eligibility[feature])<.25 else dict(feature=feature,eligibility=eligibility[feature],sign=int(eligibility[feature]>0))
    return steps,choice


def fit(images,emit):
    candidates=[]
    for gain in GAINS:
        per=[]
        for neuron in range(4):
            counts=[];empty=0
            for image in images:
                steps,_=trajectory(image['schedule'],image['voltages'][neuron],gain)
                for step in steps:emit('fit_candidate_lif_step',image_ordinal=image['ordinal'],neuron=neuron,**step)
                counts.append(sum(s['post'] for s in steps))
                empty+=sum(s['post'] for s in steps if not any(s['pre']))
            eligible=8<=sum(counts)<=64 and sum(n>0 for n in counts)>=4 and max(counts)<=8 and empty==0
            per.append(dict(neuron=neuron,post_counts=counts,total_posts=sum(counts),active_images=sum(n>0 for n in counts),
                            empty_input_posts=empty,eligible=eligible))
        candidate=dict(gain=gain,per_neuron=per);candidates.append(candidate);emit('gain_candidate',**candidate)
    gains=[]
    for neuron in range(4):
        options=[(c['gain'],c['per_neuron'][neuron]) for c in candidates if c['per_neuron'][neuron]['eligible']]
        if not options:break
        gains.append(min(options,key=lambda x:(abs(x[1]['total_posts']-32),x[0]))[0])
    return dict(gains=gains if len(gains)==4 else None,candidates=candidates,
                selection='per logical neuron min(abs(totalfitposts-32),gain)',labels_used=False)
