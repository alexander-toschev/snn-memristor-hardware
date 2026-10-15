"""Independent offline interval math. No runtime/native imports or hardware access."""
import math
from statistics import mean,stdev

CELLS=((0,8),(3,8),(5,8))
TARGETS={(0,8):(.040,.045,.050),(3,8):(.040,.045,.050),(5,8):(.030,.035,.040)}
MAX_DECISIONS=9*(271+271)
MAX_IMPACTS=9*(1+271+5+271+3*(5+30+5+3)+3)


def expected_band(target,sum_gain=330.0):
    if not math.isfinite(target) or target<=0:raise ValueError('Expected positive finite target')
    return dict(q_target=target,q_min=target*.99,q_max=target*1.01,
        r_min=sum_gain/(target*1.01),r_max=sum_gain/(target*.99))


def inside(r,band):
    return type(r) in (int,float) and math.isfinite(r) and band['r_min']<=r<=band['r_max']


def summarize(values):
    return dict(n=len(values),mean=mean(values),sd=stdev(values) if len(values)>1 else None,
        minimum=min(values),maximum=max(values))


def screen_adc(zero,low,high):
    """Recompute historical two-level screen from five independent saved samples per level."""
    if not all(len(v)==5 for v in (zero,low,high)):raise ValueError('Expected five samples per level')
    z,l,h=(summarize(v) for v in (zero,low,high))
    sl,sh=l['mean']-z['mean'],h['mean']-z['mean']
    sdl,sdh=math.hypot(l['sd'],z['sd']),math.hypot(h['sd'],z['sd'])
    snrl,snrh=sl/max(1,sdl),sh/max(1,sdh)
    cv=sdh/sh if sh>0 else None
    epsilon=h['mean']-2*l['mean']+z['mean']
    se=math.sqrt(h['sd']**2/5+4*l['sd']**2/5+z['sd']**2/5)
    tol=max(.05*abs(sh),3*se,2)
    return dict(signal_low_adc=sl,signal_high_adc=sh,snr_low=snrl,snr_high=snrh,
        cv_high=cv,linearity_epsilon_adc=epsilon,linearity_tolerance_adc=tol,
        passes=sl>0 and sh>0 and snrl>=5 and snrh>=5 and cv is not None and cv<=.05 and abs(epsilon)<=tol)


def readback_inside(readings,band,expected_count):
    return len(readings)==expected_count and all(inside(r,band) for r in readings)
