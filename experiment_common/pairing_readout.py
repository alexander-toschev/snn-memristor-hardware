"""Bounded public readout and label-free maximum-cardinality differential pairing."""
from __future__ import annotations
from collections import defaultdict
from collections.abc import Callable
import contextlib
from dataclasses import asdict, dataclass
from functools import lru_cache
import itertools
import json
import math
from pathlib import Path
import statistics
import time
from typing import TYPE_CHECKING

from .block_readout_validation import Block, Collector, Leg, Pair, patterns
from .durable import Writer, atomic_json, sha
from .native_output import DriverOutput, safe
from .selective_metrology import Converters, Reading, Source, number, object_dict
from .tiled_transfer import Profile
if TYPE_CHECKING:
    from manager.board import Connector

DELTAS = (.025,.05,.075)
SPEC: dict[str,object] = dict(schema='read-only-optimized-pairing-v1',maximum_calls=100000,
    maximum_seconds=1200,maximum_pulses=0,raw_bytes=512*1024**2,reserve_bytes=2*1024**3,
    basis_repeats=3,heldout_repeats=3,samples_per_level=4,bias_v=.025,deltas_v=DELTAS,
    maximum_voltage_v=.1,basis_planned_calls=64548,maximum_heldout_calls=27720,
    maximum_planned_calls=92268,current_limit_a=.005,predicted_headroom_v=3.3,measured_headroom_v=4.,
    selection='maximum cardinality positive historical G overlap, then sum edge scores; no labels',
    edge_score='log1p((.6*common_G_span*native_delta*sum_gain/nominal_ADC_step)/noise_penalty)',
    noise_penalty='max(1,hypot(legSDs))+maxAbsDriftPlus+maxAbsDriftMinus+linearityResidualPlus+linearityResidualMinus; +5 per leg if fewer than3 safe amplitudes',
    linearity='basis-only least-squares through zero across mean ADC responses versus native decoded increments',
    matching_ties='edge scores rounded12 decimals; max pair count, max sum score, lexicographically smallest WL pairs',
    amplitude_ties='max global pair count, max sum score, then lower requested delta',
    grouping='greedy same-WL edges by descending shared G window, then WL/BL coordinates; at most31 two-input merges',
    no_multiinput='keep all selected pairs and flag absence; never silently discard cells',
    heldout='map frozen before independent3 repeats; all nonempty masks; no refit or map filtering',
    current_state_only=True,programming_guaranteed=False,mnist_qualified=False,
    low_current_weight_is_exclusion=False,test_labels_used=False)


class WideCollector(Collector):
    """Read-only extension to .1V; original Collector and historic limits unchanged."""
    def __init__(self,connector: Connector,cv: Converters,writer: Writer,source: Source,
                 *,maximum_calls: int=100000,maximum_seconds: float=1200.) -> None:
        super().__init__(connector,cv,writer,source)
        self.maximum_calls,self.maximum_seconds=maximum_calls,maximum_seconds

    def call(self,command: dict[str,int|float|list[float]],**metadata: object) -> Reading:
        if self.calls>=self.maximum_calls or time.monotonic()-self.started>=self.maximum_seconds:
            raise TimeoutError('Finite wide readout budget')
        mode,wl=command.get('mode_flag'),command.get('wl')
        if type(wl) is not int or not isinstance(wl,int) or not 0<=wl<8:
            raise ValueError('Invalid WL')
        if mode==7:
            bl=command.get('bl')
            if type(bl) is not int or not isinstance(bl,int) or not 0<=bl<32 or command.get('vol')!=0. or command.get('sign')!=0 or command.get('pulse_width')!=0.:
                raise ValueError('Only zero-write mode7 reads permitted')
        elif mode==10:
            vol=command.get('vol')
            if not isinstance(vol,list) or len(vol)!=32 or any(v not in (0.,.025,.05,.075,.1) for v in vol):
                raise ValueError('Outside fixed read-only voltage envelope')
        else:
            raise ValueError('Only public read-only mode7/mode10')
        self.calls+=1
        ticket={**command,'id':self.calls if mode==7 else 0}
        self.emit('request',task_id=self.calls,command_si=ticket,metadata=metadata)
        values=ticket['vol']
        copied={**ticket,'vol':list(values) if isinstance(values,list) else values}
        with contextlib.redirect_stdout(DriverOutput(self,'stdout')),contextlib.redirect_stderr(DriverOutput(self,'stderr')):
            result=self.connector.impact(copied)
        self.emit('result',task_id=self.calls,command_si=ticket,native_result=safe(result))
        if not isinstance(result,(tuple,list)) or len(result)!=3:
            raise ValueError('Invalid native response')
        resistance,echo,adc=result
        if type(echo) is not int or echo!=ticket['id'] or type(adc) is not int or not 0<=adc<=16382:
            raise ValueError('Invalid native ADC/ID')
        r=number(resistance)
        if mode==7 and (r<=1e-8 or adc<20):
            raise ValueError('Invalid native resistance/ADC')
        if mode==10:
            voltage=self.cv.convert_adc_to_volt(1,14,5,adc)
            if not math.isfinite(voltage) or voltage>4.:
                raise ValueError('Measured ADC headroom')
        return Reading(task_id=self.calls,resistance_ohm=r,adc=adc)


@dataclass(frozen=True)
class Cell:
    wl: int
    bl: int
    low_g: float
    high_g: float


@dataclass(frozen=True)
class Edge:
    plus: int
    minus: int
    bl: int
    low_g: float
    high_g: float
    score: float
    expected_span_adc: float
    noise_penalty_adc: float


def candidates(path: Path) -> tuple[Cell,...]:
    payload=object_dict(json.loads(path.read_text(encoding='utf-8')))
    rows=payload['cells']
    if not isinstance(rows,list):
        raise ValueError('Pinned candidate list required')
    result=[]
    for item in rows:
        row=object_dict(item)
        cell=Cell(int(number(row['wl'])),int(number(row['bl'])),number(row['low_g_s']),number(row['high_g_s']))
        if not 0<=cell.wl<8 or not 0<=cell.bl<32 or not 0<cell.low_g<cell.high_g:
            raise ValueError('Invalid historical candidate range')
        result.append(cell)
    if len({(c.wl,c.bl) for c in result})!=len(result):
        raise ValueError('Duplicate candidate')
    return tuple(sorted(result,key=lambda c:(c.bl,c.wl)))


def maximum_matching(nodes: tuple[int,...],edges: dict[tuple[int,int],Edge]) -> tuple[Edge,...]:
    """Exact matching over at most8 WL; primary cell retention is unconditional."""
    def rank(value: tuple[Edge,...]) -> tuple[int,float,tuple[tuple[int,int],...]]:
        return -len(value),-round(sum(e.score for e in value),12),tuple((e.plus,e.minus) for e in value)
    @lru_cache(maxsize=None)
    def solve(remaining: tuple[int,...]) -> tuple[Edge,...]:
        if not remaining:
            return ()
        first,*rest=remaining
        options=[solve(tuple(rest))]
        for other in rest:
            edge=edges.get((first,other))
            if edge is not None:
                options.append((edge,*solve(tuple(v for v in rest if v!=other))))
        return min(options,key=rank)
    return solve(tuple(sorted(nodes)))


def basis_stats(basis: dict[tuple[int,int,float],list[Leg]],cell: Cell,delta: float) -> tuple[float,float,float,float]|None:
    rows=basis.get((cell.wl,cell.bl,delta),[])
    if len(rows)!=3:
        return None
    values=[r.centered('mean4') for r in rows]
    amplitudes=[]
    for level in DELTAS:
        other=basis.get((cell.wl,cell.bl,level),[])
        if len(other)==3:
            native=other[0].signal_decoded_v[cell.bl]-other[0].baseline_decoded_v[cell.bl]
            amplitudes.append((native,statistics.mean(r.centered('mean4') for r in other)))
    slope=sum(x*y for x,y in amplitudes)/sum(x*x for x,_ in amplitudes)
    native=rows[0].signal_decoded_v[cell.bl]-rows[0].baseline_decoded_v[cell.bl]
    residual=abs(statistics.mean(values)-slope*native)+(5. if len(amplitudes)<3 else 0.)
    return statistics.stdev(values),max(abs(r.drift('mean4')) for r in rows),residual,native


def choose_map(cells: tuple[Cell,...],basis: dict[tuple[int,int,float],list[Leg]],
               source_min: float,source_max: float,adc_step: float,sum_gain: float) -> dict[str,object]:
    if not 0<source_min<source_max or adc_step<=0 or sum_gain<=0:
        raise ValueError('Invalid fixed conversion/target scale')
    alternatives: list[tuple[float,tuple[Edge,...]]]=[]
    for delta in DELTAS:
        matching_edges: list[Edge]=[]
        for bl in range(32):
            local=tuple(c for c in cells if c.bl==bl)
            edges: dict[tuple[int,int],Edge]={}
            for a,b in itertools.combinations(local,2):
                low,high=max(a.low_g,b.low_g),min(a.high_g,b.high_g)
                pa,pb=basis_stats(basis,a,delta),basis_stats(basis,b,delta)
                if high<=low or pa is None or pb is None:
                    continue
                noise=max(1.,math.hypot(pa[0],pb[0]))+pa[1]+pb[1]+pa[2]+pb[2]
                expected=.6*(high-low)*pa[3]*sum_gain/adc_step
                edges[a.wl,b.wl]=Edge(a.wl,b.wl,bl,low,high,round(math.log1p(expected/noise),12),expected,noise)
            matching_edges.extend(maximum_matching(tuple(c.wl for c in local),edges))
        alternatives.append((delta,tuple(matching_edges)))
    delta,chosen=min(alternatives,key=lambda row:(-len(row[1]),-round(sum(e.score for e in row[1]),12),row[0]))
    if not chosen:
        raise ValueError('No safe positive-overlap pair, no map fabricated')
    possibilities=[]
    for edge_a,edge_b in itertools.combinations(chosen,2):
        if (edge_a.plus,edge_a.minus)!=(edge_b.plus,edge_b.minus):
            continue
        low,high=max(edge_a.low_g,edge_b.low_g),min(edge_a.high_g,edge_b.high_g)
        if high>low:
            possibilities.append((-(high-low),edge_a.plus,edge_a.minus,min(edge_a.bl,edge_b.bl),max(edge_a.bl,edge_b.bl),edge_a,edge_b))
    occupied: set[tuple[int,int,int]]=set()
    groups: list[tuple[Edge,...]]=[]
    for _,_,_,_,_,edge_a,edge_b in sorted(possibilities,key=lambda row:row[:5]):
        ka,kb=(edge_a.plus,edge_a.minus,edge_a.bl),(edge_b.plus,edge_b.minus,edge_b.bl)
        if len(groups)>=31:
            break
        if ka not in occupied and kb not in occupied:
            groups.append((edge_a,edge_b));occupied.update((ka,kb))
    multi=len(groups)
    groups.extend((e,) for e in chosen if (e.plus,e.minus,e.bl) not in occupied)
    groups.sort(key=lambda group:(group[0].plus,group[0].minus,tuple(sorted(e.bl for e in group))))
    selected: list[dict[str,object]]=[]
    for group in groups:
        low,high=max(e.low_g for e in group),min(e.high_g for e in group)
        scale=.6*(high-low)/(source_max-source_min)
        selected.append(dict(block=f'P{len(selected)+1:03d}',wl_pair=[group[0].plus,group[0].minus],
            bls=sorted(e.bl for e in group),mode='fixed_positive',delta_v=delta,global_bias=True,
            require_resolved_legs=False,conductance_scale=scale,conductance_offset=low+.2*(high-low)-scale*source_min,
            measured_common_g_s=[low,high],encoded_target_g_s=[low+.2*(high-low),low+.8*(high-low)],
            qualification='exploratory historical-window encoding; post-write MVM still required'))
    patterns_count=len(chosen)+multi
    selected_cells={(w,e.bl) for e in chosen for w in (e.plus,e.minus)}
    return dict(selected_blocks=selected,physical_cells=2*len(chosen),coefficient_slots=len(chosen),
        historical_candidate_count=len(cells),maximum_same_BL_cardinality=2*sum(sum(c.bl==bl for c in cells)//2 for bl in range(32)),
        unpaired_reserve=[dict(wl=c.wl,bl=c.bl,reason='unmatched reserve, not a defective-cell verdict') for c in cells if (c.wl,c.bl) not in selected_cells],
        edges=[asdict(e) for e in chosen],amplitude_alternatives=[dict(delta_v=d,pairs=len(v),score=sum(e.score for e in v)) for d,v in alternatives],
        selected_delta_v=delta,nonempty_patterns=patterns_count,multiinput_blocks=multi,no_analog_multiinput_group=multi==0,
        read_calls_per_load=patterns_count*8*2*44,prewrite_guard_calls_max=256,loads=(1600+len(chosen)-1)//len(chosen),
        selection_protocol=SPEC,nominal_ADC_step_v=adc_step,expected_span_is_proxy_not_measured_MVM=True,
        test_labels_used=False,mnist_qualified=False,programming_guaranteed=False)


def wide_profiles(path: Path) -> tuple[Profile,...]:
    """Separate parser preserves historic profile acceptance/source hashes."""
    data=object_dict(json.loads(path.read_text(encoding='utf-8')))
    rows=data['selected_blocks']
    if not isinstance(rows,list):
        raise ValueError('Selected blocks required')
    used: set[tuple[int,int]]=set()
    result=[]
    for item in rows:
        row=object_dict(item);wls=row['wl_pair'];bls=row['bls']
        if not isinstance(wls,list) or len(wls)!=2 or not isinstance(bls,list) or not 1<=len(bls)<=2:
            raise ValueError('Expected differential pair and1/2 BL')
        block=Block(str(row['block']),int(number(wls[0])),int(number(wls[1])),tuple(int(number(b)) for b in bls))
        delta=number(row['delta_v']);scale=number(row['conductance_scale']);offset=number(row['conductance_offset'])
        if block.plus_wl==block.minus_wl or delta not in DELTAS or not math.isfinite(scale) or scale<=0 or not math.isfinite(offset):
            raise ValueError('Invalid fixed profile')
        if row['mode']!='fixed_positive' or row['global_bias'] is not True:
            raise ValueError('Only global positive bias')
        for w in (block.plus_wl,block.minus_wl):
            for b in block.bls:
                if not 0<=w<8 or not 0<=b<32 or (w,b) in used:
                    raise ValueError('Duplicate/outside cell')
                used.add((w,b))
        result.append(Profile(block,True,delta,True,scale,offset,False))
    if not result or sum((1<<len(p.block.bls))-1 for p in result)*8*2*44+256>74518:
        raise ValueError('Empty map or transfer read budget exceeded')
    return tuple(result)


def diagnose(reader: WideCollector,cells: tuple[Cell,...],out: Path,source_min: float,
             source_max: float,adc_step: float,sum_gain: float,progress: Callable[[str],None]) -> dict[str,object]:
    basis: dict[tuple[int,int,float],list[Leg]]=defaultdict(list)
    snapshots: list[dict[str,object]]=[]
    skipped: list[dict[str,object]]=[]
    for repeat in range(3):
        for index,cell in enumerate(cells if repeat%2==0 else tuple(reversed(cells))):
            block=Block(f'C{cell.wl}_{cell.bl}',cell.wl,cell.wl,(cell.bl,))
            for delta in DELTAS if repeat%2==0 else tuple(reversed(DELTAS)):
                try:
                    leg=reader.measure_leg(block,cell.wl,'basis',repeat,'fixed_positive',delta,(cell.bl,),global_bias=True)
                except ValueError as exc:
                    if str(exc)!='Fresh summed current/headroom guard':
                        raise
                    skipped.append(dict(phase='basis',wl=cell.wl,bl=cell.bl,delta_v=delta,repeat=repeat,reason=str(exc)))
                    reader.emit('guard_skipped',**skipped[-1]);continue
                basis[cell.wl,cell.bl,delta].append(leg)
                row: dict[str,object]=dict(wl=cell.wl,bl=cell.bl,delta_v=delta,repeat=repeat,leg=asdict(leg))
                snapshots.append(row);reader.emit('basis_bracket',**row)
            if index%16==0 or index+1==len(cells):
                reader.writer.checkpoint(phase='basis',repeat=repeat,cells=index+1,impacts=reader.calls)
                progress(f'Basis {repeat+1}/3 cells {index+1}/{len(cells)}; calls {reader.calls}')
        atomic_json(out/'basis.json',snapshots)
        atomic_json(out/'guard_skips.json',skipped)
    selected=choose_map(cells,basis,source_min,source_max,adc_step,sum_gain)
    selected.update(basis_sha256=sha(out/'basis.json'),source_session=out.name,diagnostic_smoke=len(cells)!=163)
    atomic_json(out/'selected_map.json',selected)
    frozen_map_sha=sha(out/'selected_map.json')
    reader.emit('selection_frozen',map_sha256=frozen_map_sha,basis_sha256=selected['basis_sha256'])
    profiles=wide_profiles(out/'selected_map.json')
    held: list[Pair]=[]
    for repeat in range(3):
        for profile in profiles if repeat%2==0 else tuple(reversed(profiles)):
            block=profile.block
            for targets in patterns(block):
                order=(block.plus_wl,block.minus_wl) if repeat%2==0 else (block.minus_wl,block.plus_wl)
                legs: dict[int,Leg]={}
                try:
                    for wl in order:
                        legs[wl]=reader.measure_leg(block,wl,'validation',repeat,'fixed_positive',profile.delta,targets,global_bias=True)
                except ValueError as exc:
                    if str(exc)!='Fresh summed current/headroom guard':
                        raise
                    skipped.append(dict(phase='validation',block=block.name,targets=list(targets),repeat=repeat,reason=str(exc)))
                    reader.emit('guard_skipped',**skipped[-1]);continue
                pair=Pair(block.name,'validation',repeat,'fixed_positive',profile.delta,targets,order,legs[block.plus_wl],legs[block.minus_wl])
                held.append(pair);reader.emit('heldout_pair',**asdict(pair))
            reader.writer.checkpoint(phase='validation',repeat=repeat,block=block.name,impacts=reader.calls)
            progress(f'Heldout {repeat+1}/3 {block.name}; calls {reader.calls}')
    atomic_json(out/'validation.json',[asdict(row) for row in held])
    atomic_json(out/'guard_skips.json',skipped)
    diagnostics=[]
    for profile in profiles:
        block=profile.block
        for targets in patterns(block):
            rows=[r for r in held if r.block==block.name and r.targets==targets]
            if len(rows)!=3:
                diagnostics.append(dict(block=block.name,targets=list(targets),complete=False,strict_pass=False,reason='current guard prevented complete heldout'));continue
            predicted=[sum(statistics.mean(r.centered('mean4') for r in basis[wl,bl,profile.delta]) for bl in targets) for wl in (block.plus_wl,block.minus_wl)]
            actual=[r.plus.centered('mean4')-r.minus.centered('mean4') for r in rows]
            residual=abs(statistics.mean(actual)-(predicted[0]-predicted[1]))
            limit=max(5.,.1*(abs(predicted[0])+abs(predicted[1])))
            drift=max(abs(leg.drift('mean4')) for row in rows for leg in (row.plus,row.minus))
            signed_drift=max(abs(r.plus.drift('mean4')-r.minus.drift('mean4')) for r in rows)
            sd=statistics.stdev(actual)
            leg_residuals=[abs(statistics.mean((r.plus if i==0 else r.minus).centered('mean4') for r in rows)-predicted[i]) for i in (0,1)]
            leg_pass=all(error<=max(5.,.1*abs(predicted[i])) for i,error in enumerate(leg_residuals))
            passed=residual<=limit and drift<=5 and signed_drift<=5 and sd<=5 and leg_pass
            diagnostics.append(dict(block=block.name,targets=list(targets),complete=True,strict_pass=passed,
                expected_from_measured_singletons_adc=predicted,signed_mean_adc=statistics.mean(actual),signed_sd_adc=sd,
                signed_residual_adc=residual,limit_adc=limit,leg_residuals_adc=leg_residuals,
                maximum_leg_drift_adc=drift,maximum_signed_drift_adc=signed_drift))
    atomic_json(out/'heldout_checks.json',diagnostics)
    if sha(out/'selected_map.json')!=frozen_map_sha:
        raise ValueError('Heldout changed frozen map')
    return dict(candidates=len(cells),physical_cells=selected['physical_cells'],coefficient_slots=selected['coefficient_slots'],
        multiinput_blocks=selected['multiinput_blocks'],no_analog_multiinput_group=selected['no_analog_multiinput_group'],
        selected_delta_v=selected['selected_delta_v'],selected_map_sha256=frozen_map_sha,read_calls_per_load=selected['read_calls_per_load'],
        heldout_patterns=len(diagnostics),strict_pass_patterns=sum(r['strict_pass'] is True for r in diagnostics),
        guard_skips=len(skipped),pulses=0,heldout_changed_selection=False,mnist_qualified=False,
        warning='Current-state evidence only; historical G windows do not guarantee future programming')
