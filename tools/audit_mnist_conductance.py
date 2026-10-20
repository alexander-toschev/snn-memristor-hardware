"""Independent offline replay of the physical MNIST32->8 pilot; never opens a board."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC_DOCUMENT = 'docs/plans/MNIST_PHYSICAL_PILOT128_20261019.md'
SPEC_DOCUMENT_SHA = '3a2e4b6a3fce91f4fb57701d0bc200977746bbc54e0012176ea4e8d2ad1a2455'
SOURCE_IDX_SHA = '440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609'
GAINS = (1, 2, 4, 8, 10, 20, 40, 80, 100, 200, 400, 800, 1600)
GROUPS = {'train': list(range(128)), 'fit': list(range(11000,11032)), 'probe': list(range(12000,12016))}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            result.update(block)
    return result.hexdigest()


def local(name, root=ROOT):
    p, w = Path(name), PureWindowsPath(name)
    require(bool(name) and not p.is_absolute() and not w.drive and not w.root and '..' not in p.parts and '..' not in w.parts, 'Repository-relative path required')
    answer = (root / p).resolve()
    require(answer.is_relative_to(root.resolve()), 'Path escapes root')
    return answer


def same(actual, expected, label='replay'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), label + ': fields differ')
        for key, value in expected.items():
            same(actual[key], value, label + '.' + str(key))
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), label + ': length differs')
        for index, (a, b) in enumerate(zip(actual, expected)):
            same(a, b, label + '[' + str(index) + ']')
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        require(type(actual) is type(expected) and actual == expected, label + ': value/type differs')
    elif isinstance(expected, (int, float)):
        require(type(actual) in (int, float) and math.isfinite(actual) and math.isfinite(expected) and abs(actual-expected) <= 1e-12, label + ': numeric mismatch')
    else:
        raise TypeError('Unsupported replay type')



def exact_command(actual, expected):
    """Native commands have exact discrete addresses and values, unlike LIF floats."""
    require(type(actual) is dict and set(actual) == set(expected), 'Command schema differs')
    require(actual == expected, 'Exact command value differs')
    for key in ('mode_flag', 'id', 'wl', 'bl', 'sign'):
        if key in expected:
            require(type(actual[key]) is int, 'Command integer type differs: '+key)
    for key in ('vol', 'pulse_width'):
        if key in expected:
            values = actual[key] if isinstance(expected[key], list) else [actual[key]]
            if isinstance(expected[key], list):
                require(type(actual[key]) is list, 'Command vector type differs')
            require(all(type(value) in (int,float) and math.isfinite(value) for value in values), 'Command numeric type differs')


def unknown_physical_duration(manifest, start):
    for row in (manifest, start):
        require('physical_pulse_duration_s' in row and row['physical_pulse_duration_s'] is None,
                'Physical pulse duration is unknown, not measured')


def schedule_from_pixels(pixels):
    require(isinstance(pixels, list) and len(pixels) == 784 and all(type(x) is int and 0 <= x <= 255 for x in pixels), 'Invalid image pixels')
    ys, xs = (0,7,14,21,28), (0,3,7,10,14,17,21,24,28)
    blocks, schedule = [], [[] for _ in range(32)]
    for row in range(4):
        for column in range(8):
            bl = 8*row+column
            values = [pixels[y*28+x] for y in range(ys[row],ys[row+1]) for x in range(xs[column],xs[column+1])]
            total, area = sum(values), len(values)
            tick = min(31, (32*(255*area-total))//(255*area)) if total else None
            blocks.append(dict(bl=bl, sum_uint8=total, area=area, spike_t=tick))
            if tick is not None:
                schedule[tick].append(bl)
    return blocks, schedule


def verify_fixture(fixture):
    require(fixture['schema'] == 'mnist-conductance-label-free176-v1' and fixture['source_gzip_sha256'] == SOURCE_IDX_SHA, 'Wrong fixture provenance')
    same(fixture['groups'], GROUPS, 'fixture groups')
    require(fixture['split'] == 'train' and fixture['image_count'] == 176, 'Wrong fixture split/count')
    for key in ('labels_included', 'labels_read', 'test_split_read', 'model_executed', 'board_access'):
        require(fixture.get(key) is False, 'Disallowed fixture claim: '+key)
    order = GROUPS['train']+GROUPS['fit']+GROUPS['probe']
    require([item['train_index'] for item in fixture['images']] == order, 'Fixture order/coverage differs')
    schedules, joined = {}, hashlib.sha256()
    for row in fixture['images']:
        require(row['shape'] == [28,28], 'Image shape differs')
        blocks, schedule = schedule_from_pixels(row['pixels_uint8'])
        data = bytes(row['pixels_uint8'])
        require(hashlib.sha256(data).hexdigest() == row['image_sha256'], 'Image SHA differs')
        same(row['blocks'], blocks, 'pooled integer blocks')
        same(row['schedule'], schedule, 'TTFS schedule')
        require(digest(schedule) == row['schedule_sha256'], 'Schedule SHA differs')
        joined.update(data)
        schedules[row['train_index']] = schedule
    require(joined.hexdigest() == fixture['images_sha256'], 'Combined image SHA differs')
    return schedules


def replay_neuron(schedule, voltages, gain):
    require(len(schedule) == len(voltages) == 32 and gain in GAINS, 'LIF shape/gain differs')
    u, pre_trace, post_trace, eligibility, steps = 0., [0.]*32, 0., [0.]*32, []
    decay = math.exp(-.25)
    for tick in range(32):
        require(math.isfinite(voltages[tick]), 'Nonfinite measured voltage')
        before = u
        unreset = .5*u+gain*voltages[tick]
        post = int(unreset >= 1.)
        u = 0. if post else unreset
        pre = [int(bl in schedule[tick]) for bl in range(32)]
        pre_trace = [value*decay for value in pre_trace]
        post_trace *= decay
        increments = [value*post-1.05*post_trace*spike for value,spike in zip(pre_trace,pre)]
        eligibility = [value+increment for value,increment in zip(eligibility,increments)]
        pre_trace = [value+spike for value,spike in zip(pre_trace,pre)]
        post_trace += post
        steps.append(dict(t=tick, gain=gain, u_before=before, u_unreset=unreset, u_after=u,
            post=post, pre=pre, voltage_v=voltages[tick], eligibility=list(eligibility),
            eligibility_increment=increments, pre_trace=list(pre_trace), post_trace=post_trace))
    return steps


def select_gain(fit_inputs):
    """Independent train-fit-only replay; input is 32 (schedule, four-WL voltages) pairs."""
    require(len(fit_inputs) == 32, 'Exactly32 fit images required')
    candidates = []
    for gain in GAINS:
        rows = []
        for wl in range(4):
            trajectories = [replay_neuron(schedule, volts[wl], gain) for schedule,volts in fit_inputs]
            counts = [sum(step['post'] for step in trajectory) for trajectory in trajectories]
            zero_posts = sum(step['post'] for trajectory in trajectories for step in trajectory if not any(step['pre']))
            eligible = 8 <= sum(counts) <= 64 and sum(n > 0 for n in counts) >= 4 and max(counts) <= 8 and zero_posts == 0
            rows.append(dict(wl=wl, posts=sum(counts), active_images=sum(n > 0 for n in counts), max_posts=max(counts), zero_posts=zero_posts, eligible=eligible, trajectories=trajectories))
        total = sum(row['posts'] for row in rows)
        candidates.append(dict(gain=gain, eligible=all(row['eligible'] for row in rows), total_posts=total, objective=[abs(total-128),gain], rows=rows))
    eligible = [candidate for candidate in candidates if candidate['eligible']]
    gain = min(eligible,key=lambda row:row['objective'])['gain'] if eligible else None
    return gain, candidates


def choose_update(eligibility):
    require(len(eligibility) == 32 and all(math.isfinite(x) for x in eligibility), 'Invalid eligibility')
    bl = max(range(32), key=lambda index: (abs(eligibility[index]),-index))
    if abs(eligibility[bl]) < .25:
        return None
    return dict(bl=bl, eligibility=eligibility[bl], sign=int(eligibility[bl] > 0))


def durable_records(session):
    """Stream verified closed gzip chunks; exact completion, no partial-tail salvage."""
    directory = session/'raw'
    index = json.loads((directory/'index.json').read_text(encoding='utf-8'))
    require(index.get('schema') == 'lossless-chunked-jsonl-v1' and index.get('stream_finalized') is True, 'Raw stream not finalized')
    wanted = {'index.json'} | {part['path'] for part in index['parts']}
    require({p.name for p in directory.iterdir()} == wanted, 'Raw directory inventory differs')
    joined, count, size = hashlib.sha256(), 0, 0
    for number, part in enumerate(index['parts']):
        require(part['path'] == 'part-%06d.jsonl.gz'%number, 'Noncontinuous raw chunk names')
        path = directory/part['path']
        require(0<part['bytes']<=8*1024**2 and 0<part['records'] and part['compressed_bytes']<=128*1024**2, 'Raw chunk bound exceeded')
        require(path.stat().st_size == part['compressed_bytes'] and sha(path) == part['compressed_sha256'], 'Compressed raw SHA/size differs')
        digest_part, part_size, lines = hashlib.sha256(), 0, 0
        with gzip.open(path,'rb') as stream:
            for line in iter(lambda: stream.readline(8*1024**2+1), b''):
                require(line.endswith(b'\n') and len(line) <= 8*1024**2, 'Truncated/oversized JSONL record')
                digest_part.update(line);joined.update(line)
                part_size += len(line);size += len(line);lines += 1;count += 1
                require(size <= 2*1024**3, 'Raw expansion exceeds maximum')
                yield json.loads(line)
        require(part_size == part['bytes'] and lines == part['records'] and digest_part.hexdigest() == part['sha256'], 'Expanded chunk SHA/size/count differs')
    require(count == index['records'] and joined.hexdigest() == index['sha256'], 'Whole raw SHA/count differs')

def expected_protocol():
    return dict(schema='mnist-pooled32-conductance-pilot128-v1', timesteps=32,
        shape=[32,8], learning_wl=[0,1,2,3], control_wl=[4,5,6,7], groups=GROUPS,
        amplitude_v=.05, maximum_mask=4, reset_v=1.4, set_v=1.7,
        preparation_signs=[0]*4+[1]*4, seed=20261019, gain_candidates=[float(g) for g in GAINS],
        gain_fit='learning WL only; each8..64posts,>=4active images,max8/image,no empty-input posts',
        gain_objective='min(abs(total learning posts-128),gain)', lif_leak=.5, lif_threshold=1.,
        trace_decay=math.exp(-.25), depression=1.05, eligibility_threshold=.25,
        updates='one max-absolute-E BL per learningWL after complete image/zero_after; lowerBL tie',
        checkpoints=14, checkpoint_calls=3840, retention_seconds=60.,
        maximum_impacts=172288, maximum_pulses=2560, maximum_decisions=2560,
        no_eligible_maximum_impacts=34560, no_eligible_maximum_pulses=2048,
        simulator_pulse_width_s=.0001, hardware_submitted_width_s=0., hardware_physical_width_s=None,
        maximum_raw_expanded_bytes=2*1024**3, raw_chunk_bytes=8*1024**2,
        raw_minimum_free_bytes=2*1024**3, initial_minimum_free_bytes=6*1024**3,
        labels_used=False, classification_accuracy_measured=False, reference_sim_enabled=True,
        reference_maximum_impacts=171776, reference_maximum_pulses=2048,
        reference_semantics='SIM only: same initial array/preparation/inputs, main gain frozen; own shadow choices,4reads,no training pulse')


class Replay:
    """Protocol-driven event consumer. Every native request is consumed exactly once."""
    def __init__(self, records, fixture, source, reference=None):
        require(source in ('hardware','simulator'), 'Unknown source')
        require(reference is None or source=='simulator', 'Reference must be SIM only')
        self.reference = reference
        self.rows, self.source = iter(records), source
        self.schedules = verify_fixture(fixture)
        self.images = {row['train_index']:row for row in fixture['images']}
        self.calls = self.pulses = self.decisions = self.ordinal = self.records = 0
        self.last_monotonic = None
        self.gain = self.calibration = None
        self.updates, self.outcomes, self.checkpoints = [], [], []
        self.participation = [[0]*32 for _ in range(8)]
        self.input_events = {phase:[0]*32 for phase in ('fit','preprobe','train','postprobe','retentionprobe')}
        self.mode_counts, self.pulse_counts = Counter(), Counter()
        self.max_mvm_current = 0.

    def take(self, event=None):
        from datetime import datetime
        try:
            row = next(self.rows)
        except StopIteration as exc:
            raise ValueError('Premature EOF; expected '+str(event)) from exc
        self.records += 1
        require(row.get('source') == self.source, 'Mixed raw source')
        stamp = datetime.fromisoformat(row['utc'])
        require(stamp.tzinfo is not None, 'Naive timestamp')
        clock = row['monotonic_s']
        require(type(clock) in (int,float) and math.isfinite(clock) and (self.last_monotonic is None or clock >= self.last_monotonic), 'Nonmonotonic raw chronology')
        self.last_monotonic = clock
        require(event is None or row['event'] == event, 'Unexpected event '+str(row.get('event'))+'; expected '+str(event))
        return row

    @staticmethod
    def payload(row):
        return {key:value for key,value in row.items() if key not in ('event','source','utc','monotonic_s')}

    def expect(self, event, **fields):
        row = self.take(event)
        same(self.payload(row), json.loads(json.dumps(fields)), event)
        return row

    def call(self, command, phase, **metadata):
        self.calls += 1
        require(self.calls <= (171776 if self.reference else 172288), 'Impact budget exceeded')
        task = dict(command, id=self.calls if command['mode_flag']==7 else 0)
        fields = dict(task_id=self.calls, command_si=task, phase=phase, **metadata)
        request = self.expect('request', **fields)
        exact_command(request['command_si'],task)
        require(type(request['task_id']) is int,'Request task ID type differs')
        result = self.take()
        while result['event'] == 'driver_output':
            require(set(self.payload(result)) == {'task_id','stream','text'} and result['task_id']==self.calls and result['stream'] in ('stdout','stderr') and isinstance(result['text'],str), 'Unbound driver output')
            result = self.take()
        require(result['event']=='result', 'Native call lacks result')
        p = self.payload(result)
        require(set(p)==set(fields)|{'native_result','latency_s'}, 'Native result fields differ')
        same({key:p[key] for key in fields},fields,'result request ownership')
        exact_command(p['command_si'],task)
        require(type(p['task_id']) is int,'Result task ID type differs')
        latency=p['latency_s']
        require(type(latency) in (int,float) and math.isfinite(latency) and 0 <= latency <= result['monotonic_s']-request['monotonic_s']+1e-6, 'Invalid native latency')
        native=p['native_result']
        require(isinstance(native,list) and len(native)==3, 'Invalid native tuple')
        resistance,echo,adc=native
        require(type(echo) is int and echo==task['id'] and type(adc) is int, 'Native echo/ADC type differs')
        if command['mode_flag']==7:
            require(type(resistance) in (float,int) and math.isfinite(resistance) and resistance>1e-8 and 20<=adc<=16382, 'Invalid native R/ADC')
        else:
            require(command['mode_flag']==10 and 0<=adc<=16382, 'Invalid MVM ADC')
        self.mode_counts[command['mode_flag']] += 1
        return dict(r=resistance,adc=adc,task_id=self.calls)

    def read(self, wl, bl, phase, **metadata):
        require(0 <= wl < 8 and 0 <= bl < 32, 'Read outside256 cells')
        return self.call(dict(mode_flag=7,wl=wl,bl=bl,vol=0.,sign=0,pulse_width=0.),phase,**metadata)

    def vector(self, wl, mask, phase, **metadata):
        require(mask==sorted(set(mask)) and len(mask)<=4 and all(type(b) is int and 0<=b<32 for b in mask), 'Invalid vector mask')
        guards = [self.read(wl,bl,'vector_guard',vector_phase=phase,**metadata) for bl in mask]
        amplitude=max(.05,round(int(round(.05*4095/5,0))*5/4095,3))
        current=sum(amplitude/row['r'] for row in guards)
        require(current<=.005 and 330*current<3.3, 'MVM current guard exceeded')
        self.max_mvm_current=max(self.max_mvm_current,current)
        result=self.call(dict(mode_flag=10,wl=wl,vol=[.05 if bl in mask else 0. for bl in range(32)]),phase,mask=mask,**metadata)
        voltage=round(result['adc']*5/16383,5)
        require(math.isfinite(voltage) and voltage<=4., 'MVM measured headroom exceeded')
        return dict(task_id=result['task_id'],adc=result['adc'],voltage_v=voltage,guards=guards)

    def pulse(self, wl, bl, sign, phase, **metadata):
        self.decisions+=1
        require(self.decisions<=2560 and phase in ('preparation','training_update') and (phase!='training_update' or wl<4), 'Decision budget/control write violation')
        level=1.7 if sign else 1.4
        sham=self.read(wl+4,bl,'paired_control_guard',**metadata) if phase=='training_update' else None
        guard=self.read(wl,bl,'pulse_guard',intended_phase=phase,**metadata)
        admitted=level/guard['r']<=(.002 if sign else .04)
        suppressed=self.reference is not None and phase=='training_update'
        decision=dict(wl=wl,bl=bl,sign=sign,voltage_v=level,phase=phase,decision=self.decisions,
            guard=guard,control_guard=sham,guard_admissible=admitted,admitted=admitted and not suppressed,reference_suppressed=suppressed,**metadata)
        self.expect('pulse_decision',**decision)
        if suppressed:
            post=self.read(wl,bl,'pulse_post_r',intended_phase=phase,**metadata)
            control_post=self.read(wl+4,bl,'paired_control_post_r',**metadata)
            decision.update(post=post,control_post=control_post,delta_inverse_r=1/post['r']-1/guard['r'])
        elif admitted:
            self.pulses+=1
            require(self.pulses<=(2048 if self.reference else 2560),'Pulse budget exceeded')
            pulse=self.call(dict(mode_flag=7,wl=wl,bl=bl,vol=level,sign=sign,pulse_width=.0001 if self.source=='simulator' else 0.),phase,**metadata)
            post=self.read(wl,bl,'pulse_post_r',intended_phase=phase,**metadata)
            control_post=self.read(wl+4,bl,'paired_control_post_r',**metadata) if phase=='training_update' else None
            decision.update(pulse=pulse,post=post,control_post=control_post,delta_inverse_r=1/post['r']-1/guard['r'])
            self.pulse_counts[(phase,sign)]+=1
        self.expect('pulse_complete',**decision)
        self.updates.append(decision)

    def checkpoint(self, phase):
        import statistics
        self.expect('checkpoint_begin',phase=phase)
        means=[]
        for wl in range(8):
            values=[]
            for bl in range(32):
                reads=[self.read(wl,bl,'checkpoint_r',checkpoint_phase=phase,repeat=repeat) for repeat in range(3)]
                brackets=[]
                for repeat in range(3):
                    metadata=dict(checkpoint_phase=phase,cell_bl=bl,repeat=repeat)
                    before=self.vector(wl,[],'checkpoint_zero_before',**metadata)
                    signal=self.vector(wl,[bl],'checkpoint_signal',**metadata)
                    after=self.vector(wl,[],'checkpoint_zero_after',**metadata)
                    brackets.append(dict(before=before,signal=signal,after=after,centered_v=signal['voltage_v']-(before['voltage_v']+after['voltage_v'])/2))
                self.expect('checkpoint_cell',phase=phase,wl=wl,bl=bl,reads=reads,responses=brackets)
                values.append(statistics.mean(1/read['r'] for read in reads))
            means.append(values)
        result=dict(phase=phase,mean_inverse_r=means)
        self.expect('checkpoint_complete',**result)
        self.checkpoints.append(result)

    def image(self, phase, index):
        require((phase=='fit' and self.gain is None) or (phase!='fit' and self.gain in GAINS),'Acquisition before gain freeze')
        ordinal=self.ordinal;self.ordinal+=1
        schedule=self.schedules[index]
        self.expect('image_begin',phase=phase,ordinal=ordinal,train_index=index,image_sha256=self.images[index]['image_sha256'],schedule_sha256=digest(schedule),schedule=schedule)
        metadata=dict(image_phase=phase,image_ordinal=ordinal,train_index=index)
        order=list(range(8)) if ordinal%2==0 else list(reversed(range(8)))
        before={wl:self.vector(wl,[],'image_zero_before',**metadata) for wl in order}
        signals=[]
        for tick, mask in enumerate(schedule):
            for bl in mask:self.input_events[phase][bl]+=1
            for micro, offset in enumerate(range(0,len(mask),4)):
                chunk=mask[offset:offset+4]
                wls=list(range(8)) if (ordinal+tick+micro)%2==0 else list(reversed(range(8)))
                for wl in wls:
                    signal=self.vector(wl,chunk,'image_signal',t=tick,micro=micro,**metadata)
                    signals.append(dict(wl=wl,t=tick,micro=micro,mask=chunk,**signal))
                    for bl in chunk:self.participation[wl][bl]+=1
        after={wl:self.vector(wl,[],'image_zero_after',**metadata) for wl in order[::-1]}
        voltages=[[0.]*32 for _ in range(8)]
        for signal in signals:
            wl=signal['wl']
            voltages[wl][signal['t']]+=signal['voltage_v']-(before[wl]['voltage_v']+after[wl]['voltage_v'])/2
        result=dict(phase=phase,ordinal=ordinal,train_index=index,schedule=schedule,voltages=voltages,
            zero_before=before,zero_after=after,signals=signals,zero_drift_v=[after[wl]['voltage_v']-before[wl]['voltage_v'] for wl in range(8)])
        self.expect('image_measurements_complete',**result)
        if phase!='fit':
            neurons=[]
            for wl in range(8):
                steps=replay_neuron(schedule,voltages[wl],self.gain)
                for step in steps:
                    self.expect('lif_step',image_ordinal=ordinal,image_phase=phase,train_index=index,wl=wl,**step)
                neurons.append(dict(wl=wl,posts=[s['post'] for s in steps],eligibility=steps[-1]['eligibility'],choice=choose_update(steps[-1]['eligibility'])))
            outcome=dict(phase=phase,ordinal=ordinal,train_index=index,neurons=neurons,zero_drift_v=result['zero_drift_v'])
            self.expect('image_feedback_complete',**outcome);self.outcomes.append(outcome)
            if phase=='train':
                for wl in (range(4) if index%2==0 else reversed(range(4))):
                    choice=neurons[wl]['choice']
                    if choice is None:self.expect('no_training_update',image_ordinal=ordinal,train_index=index,wl=wl)
                    else:self.pulse(wl,choice['bl'],choice['sign'],'training_update',image_ordinal=ordinal,train_index=index,eligibility=choice['eligibility'])
        self.expect('image_complete',phase=phase,ordinal=ordinal,train_index=index,impacts=self.calls,pulses=self.pulses)
        return result

    def gain_fit(self, images):
        candidates=[]
        for gain in GAINS:
            rows=[]
            for wl in range(4):
                counts=[];empty=0
                for image in images:
                    steps=replay_neuron(image['schedule'],image['voltages'][wl],gain)
                    for step in steps:
                        self.expect('fit_candidate_lif_step',image_ordinal=image['ordinal'],image_phase='fit',train_index=image['train_index'],wl=wl,**step)
                    counts.append(sum(s['post'] for s in steps))
                    empty+=sum(s['post'] for s in steps if not any(s['pre']))
                eligible=8<=sum(counts)<=64 and sum(n>0 for n in counts)>=4 and max(counts)<=8 and empty==0
                rows.append(dict(wl=wl,post_counts=counts,total_posts=sum(counts),images_with_posts=sum(n>0 for n in counts),empty_input_posts=empty,eligible=eligible))
            total=sum(row['total_posts'] for row in rows)
            candidate=dict(gain=gain,per_wl=rows,total_posts=total,eligible=all(row['eligible'] for row in rows),objective=[abs(total-128),gain])
            self.expect('gain_candidate',**candidate);candidates.append(candidate)
        eligible=[candidate for candidate in candidates if candidate['eligible']]
        self.gain=min(eligible,key=lambda row:tuple(row['objective']))['gain'] if eligible else None
        self.calibration=dict(selected_gain=self.gain,candidates=candidates,fit_image_ordinals=list(range(32)),fit_train_indices=GROUPS['fit'],control_used=False,labels_used=False)
        self.expect('gain_frozen',last_native_task_id=self.calls,**self.calibration)

    def run(self, protocol):
        same(protocol,expected_protocol(),'registered independent protocol')
        self.expect('session_begin',protocol=protocol,reference=self.reference)
        opened=self.take('port_open')
        require(set(self.payload(opened))=={'result'} and bool(opened['result']), 'Open not confirmed')
        self.checkpoint('initial')
        for decision in range(8):
            for bl in range(32):
                for wl in (range(8) if (decision+bl)%2==0 else reversed(range(8))):
                    self.pulse(wl,bl,int(decision>=4),'preparation',preparation_step=decision)
        self.checkpoint('after_preparation')
        fit=[self.image('fit',index) for index in GROUPS['fit']]
        self.checkpoint('after_fit')
        if self.reference is None:
            self.gain_fit(fit)
        else:
            self.gain=self.reference['gain']
            require(self.gain in GAINS,'Reference gain outside fixed grid')
            self.calibration=dict(selected_gain=self.gain,candidates=[],selection='bound main fit only; reference fit does not change gain',reference=self.reference,fit_image_ordinals=list(range(32)),fit_train_indices=GROUPS['fit'])
            self.expect('gain_frozen',last_native_task_id=self.calls,**self.calibration)
        if self.gain is None:
            self.expect('calibration_stop',reason='No fixed candidate eligible; no probes or training')
            require(self.calls<=34560 and self.pulses<=2048, 'No-eligible cap exceeded')
            status='uninformative_calibration'
        else:
            for index in GROUPS['probe']:self.image('preprobe',index)
            self.checkpoint('after_preprobe')
            for index in GROUPS['train']:
                self.image('train',index)
                if (index+1)%16==0:self.checkpoint('after_train_%03d'%(index+1))
            for index in GROUPS['probe']:self.image('postprobe',index)
            self.checkpoint('after_postprobe')
            begin=self.expect('retention_begin',seconds=60.)
            end=self.take('retention_end');elapsed=end['elapsed_s']
            require(set(self.payload(end))=={'elapsed_s'} and type(elapsed) in (int,float) and math.isfinite(elapsed) and elapsed>=60. and end['monotonic_s']-begin['monotonic_s']>=elapsed, 'Retention wait invalid')
            for index in GROUPS['probe']:self.image('retentionprobe',index)
            self.checkpoint('after_retentionprobe');status='completed'
        closed=self.take('port_close')
        require(set(self.payload(closed))=={'result'} and bool(closed['result']), 'Close not confirmed')
        self.expect('session_end',status=status,error=None,impacts=self.calls,pulses=self.pulses,decisions=self.decisions)
        try:
            next(self.rows)
        except StopIteration:
            pass
        else:
            raise ValueError('Events after terminal session_end')
        require(self.ordinal==(208 if status=='completed' else 32) and len(self.checkpoints)==(14 if status=='completed' else 3), 'Incomplete image/checkpoint coverage')
        return self.summary(status)

    def summary(self,status):
        return dict(status=status,impacts=self.calls,pulses=self.pulses,decisions=self.decisions,
            reference=self.reference,diagnostics=replay_diagnostics(self.outcomes,self.checkpoints,self.updates),
            measured_images=self.ordinal,gain=self.gain,calibration=self.calibration,
            image_cell_participation=self.participation,input_event_counts=self.input_events,
            outcomes=self.outcomes,updates=self.updates,checkpoints=self.checkpoints,
            classification_accuracy_measured=False,physical_lif_integrator=False,
            learning_cells=128,read_cells=256,full10k_executed=False,
            summary_interpretation='Physical conductance updates with digitally evaluated measured-input LIF/eligibility; no accuracy or intrinsic delta-t claim')

def replay_diagnostics(outcomes, checkpoints, updates):
    """Derived reporting only; no thresholds learned from validation/probe data."""
    import statistics
    phases={}
    for phase in ('preprobe','train','postprobe','retentionprobe'):
        images=[image for image in outcomes if image['phase']==phase]
        if not images:continue
        traces=[[image['neurons'][wl]['posts'] for image in images] for wl in range(8)]
        counts=[[sum(trace) for trace in row] for row in traces]
        pairs=[]
        for left in range(8):
            for right in range(left+1,8):
                joined=list(zip(traces[left],traces[right]))
                bits=[(a,b) for aa,bb in joined for a,b in zip(aa,bb)]
                union=sum(bool(a or b) for a,b in bits)
                active=[(aa,bb) for aa,bb in joined if any(aa) or any(bb)]
                pairs.append(dict(wl_pair=[left,right],exact_matching_images=sum(aa==bb for aa,bb in joined),active_images=len(active),matching_active_images=sum(aa==bb for aa,bb in active),hamming_bits=sum(a!=b for a,b in bits),jaccard=sum(bool(a and b) for a,b in bits)/union if union else None,empty_union=union==0))
        phases[phase]=dict(images=len(images),post_counts_by_wl=counts,silent_images_by_wl=[row.count(0) for row in counts],
            unique_eight_neuron_outputs=len({tuple(bit for neuron in image['neurons'] for bit in neuron['posts']) for image in images}),
            all_learning_traces_identical=all(image['neurons'][wl]['posts']==image['neurons'][0]['posts'] for image in images for wl in range(1,4)),pairwise=pairs,
            eligibility_signs_by_wl=[dict(positive=sum(e>0 for image in images for e in image['neurons'][wl]['eligibility']),negative=sum(e<0 for image in images for e in image['neurons'][wl]['eligibility'])) for wl in range(8)])
    conductance=[]
    for checkpoint in checkpoints:
        banks=checkpoint['mean_inverse_r'];pairs=[]
        for a in range(8):
            for b in range(a+1,8):
                denominator=math.sqrt(sum(v*v for v in banks[a])*sum(v*v for v in banks[b]))
                pairs.append(dict(wl_pair=[a,b],cosine=sum(x*y for x,y in zip(banks[a],banks[b]))/denominator if denominator else None))
        conductance.append(dict(phase=checkpoint['phase'],row_mean=[statistics.mean(row) for row in banks],row_cv=[statistics.pstdev(row)/statistics.mean(row) for row in banks],pairwise_cosine=pairs))
    attempts=[[[0,0] for _ in range(32)] for _ in range(8)]
    pulses=[[[0,0] for _ in range(32)] for _ in range(8)]
    train=[item for item in updates if item['phase']=='training_update']
    for item in train:
        attempts[item['wl']][item['bl']][item['sign']]+=1
        pulses[item['wl']][item['bl']][item['sign']]+=int(item['admitted'])
    deltas=[item['delta_inverse_r'] for item in train if item['admitted']]
    train_images=[image for image in outcomes if image['phase']=='train']
    learning_posts=sum(sum(image['neurons'][wl]['posts']) for image in train_images for wl in range(4))
    nonzero_e=sum(e!=0 for image in train_images for wl in range(4) for e in image['neurons'][wl]['eligibility'])
    return dict(phases=phases,conductance=conductance,training_decisions_by_cell_reset_set=attempts,training_pulses_by_cell_reset_set=pulses,
        training_attempts=len(train),training_admitted_pulses=sum(item['admitted'] for item in train),guard_skipped=sum(not item['guard_admissible'] for item in train),reference_suppressed=sum(item['reference_suppressed'] for item in train),
        mechanism=dict(learning_bank_train_posts=learning_posts,nonzero_learning_eligibility=nonzero_e,admitted_updates_with_post_r=sum(item['admitted'] and 'post' in item for item in train),uninformative_no_learning_posts=learning_posts==0,uninformative_no_eligibility=nonzero_e==0,no_admitted_training_pulses=not any(item['admitted'] for item in train)),
        admitted_delta_g_signs=dict(positive=sum(x>0 for x in deltas),negative=sum(x<0 for x in deltas),unchanged=sum(x==0 for x in deltas)))


FIXTURE_PATH='experiments/mnist_conductance_20261019/fixture.json'
FIXTURE_SHA='02a75724f9048d451f1a82aa313c225310ff0ce4bf85bd1de9e90ae660323356'
HELPER_PINS={
    'experiments/unn_rev3_20261014/paths.py':'2c841b4a446a767595c68aad243abd966cee94ffd91f74c3c2823998fdc9ec1c',
    'tools/prepare_mnist_conductance_fixture.py':'9f195d33cb27309c3ed98403fe33eac4bb945d6d4e7b434f81142f98343d8215',
    'experiments/cifar_resumable_20261019/durable.py':'b00c094da70af933a7318a119fe5439b40d957c7aaabb39c752170a1dbf2abb3',
    'experiments/conductance_stdp_20261019/core.py':'005586394be54aad0ac7d116c729d7b733628161ef7a442975ee9b94209aa86f',
    'cell_calibration.py':'eff0b2f72586f6a6659a13fba410c6ad0d564bd95d3b83e24b13cf739121c2e8',
    'read_all.py':'b30ae2f4dd3cd166dbc5cefb07b558127de2bbdba1f6674c40b1b45f4dd74d5f',
    'native_conversions.py':'2ad2e9c057bb14979b083a6e846e8c417c080a6c5273d848e31f893b7970253e',
    'experiment_hardware.py':'059dc4f827943a9df631631f52e776b57182afc9d36db0c1ad5246427f73548c'}


def verify_bundle(session, freeze_path):
    require(session.is_dir(), 'Session missing')
    manifest=json.loads((session/'manifest.json').read_text(encoding='utf-8'))
    start=json.loads((session/'start_manifest.json').read_text(encoding='utf-8'))
    protocol=json.loads((session/'protocol.json').read_text(encoding='utf-8'))
    unknown_physical_duration(manifest,start)
    same(protocol,expected_protocol(),'protocol')
    require(manifest.get('status') in ('completed','uninformative_calibration'), 'Only terminal complete/no-gain bundles qualify')
    require(not any(key in manifest for key in ('error','cleanup_error','raw_finish_error')), 'Session records runtime failure')
    for key in ('session_id','source','protocol_sha256','source_sha256','fixture_sha256','reference','invocation','environment','array_initial_sha256','started_utc'):
        same(manifest[key],start[key],'immutable manifest.'+key)
    require(start['status']=='failed' and manifest['session_id']==session.name, 'Wrong start state/session binding')
    require(manifest['protocol_sha256']==digest(protocol), 'Manifest protocol SHA differs')
    require(manifest['fixture_sha256']==FIXTURE_SHA==sha(session/'fixture.json')==sha(ROOT/FIXTURE_PATH),'Fixture byte pin differs')
    require(sha(ROOT/SPEC_DOCUMENT)==SPEC_DOCUMENT_SHA,'Reviewed design changed')
    freeze=json.loads(freeze_path.read_text(encoding='utf-8'))
    same(freeze['protocol'],protocol,'code freeze protocol')
    require(freeze['protocol_sha256']==digest(protocol), 'Code freeze protocol hash differs')
    freeze_sources=freeze['source_sha256']
    new_sources={p.relative_to(ROOT).as_posix() for p in (ROOT/'experiments/mnist_conductance_20261019').iterdir() if p.is_file() and p.suffix in ('.py','.md')}
    require(new_sources|{'tools/audit_mnist_conductance.py'} <= set(freeze_sources), 'Code freeze lacks acquisition/auditor sources')
    for relative,pin in freeze_sources.items():
        require(sha(local(relative))==pin,'Reviewed file changed: '+relative)
    for relative,pin in HELPER_PINS.items():
        require(sha(local(relative))==pin,'Pinned helper changed: '+relative)
    # Read-only source/INI verifier; no Collector or acquisition module is imported.
    sys.path[:0]=[str(ROOT),str(ROOT/'.deps')]
    from cell_calibration import check_sources
    _,snapshot,expected=check_sources(manifest['source'])
    expected.update(HELPER_PINS)
    expected.update({SPEC_DOCUMENT:SPEC_DOCUMENT_SHA,FIXTURE_PATH:FIXTURE_SHA})
    expected.update({relative:freeze_sources[relative] for relative in new_sources})
    same(manifest['source_sha256'],expected,'exact acquisition source inventory')
    for relative,pin in expected.items():
        require(sha(local(relative))==pin,'Local source bytes differ: '+relative)
    require((session/'board.ini').read_bytes()==snapshot,'Board INI differs')
    if manifest['source']=='simulator':
        from cell_calibration import SIM_INI
        require((session/'simulator_settings.ini').read_bytes()==local(SIM_INI).read_bytes(),'SIM INI snapshot differs')
        require(sha(session/'array_initial.cb')==manifest['array_initial_sha256'],'Immutable initial SIM array differs')
    actual={p.relative_to(session).as_posix() for p in session.rglob('*') if p.is_file()}-{'manifest.json'}
    require(actual==set(manifest['result_sha256']), 'Result inventory differs')
    for relative,pin in manifest['result_sha256'].items():
        require(sha(local(relative,session))==pin,'Result bytes differ: '+relative)
    require(manifest.get('classification_accuracy_measured') is False and manifest.get('full10k_executed') is False,'Overclaimed result scope')
    return manifest,protocol,json.loads((session/'fixture.json').read_text(encoding='utf-8'))


def audit_session(session, freeze_path, _reference_depth=0):
    session,freeze_path=Path(session).resolve(),Path(freeze_path).resolve()
    require(session.is_relative_to(ROOT) and freeze_path.is_relative_to(ROOT), 'Repository-local audit paths required')
    manifest,protocol,fixture=verify_bundle(session,freeze_path)
    reference=manifest['reference'];main_report=None
    if reference is not None:
        require(_reference_depth==0 and manifest['source']=='simulator','Recursive/non-SIM reference forbidden')
        require(set(reference)=={'session','manifest_sha256','observations_sha256','gain','array_initial_sha256','fit_raw_index_sha256'}, 'Reference binding incomplete')
        main=local(reference['session'])
        require(main!=session and main.parent==ROOT/'results/mnist_conductance_20261019','Invalid main reference path')
        main_manifest=json.loads((main/'manifest.json').read_text())
        require(main_manifest.get('reference') is None and main_manifest['source']=='simulator' and main_manifest['status']=='completed','Main must be a completed native SIM')
        require(main_manifest['source_sha256']==manifest['source_sha256'] and main_manifest['array_initial_sha256']==reference['array_initial_sha256']==manifest['array_initial_sha256'],'Main/reference source or initial array differs')
        for name,key in (('manifest.json','manifest_sha256'),('observations.json','observations_sha256'),('raw/index.json','fit_raw_index_sha256')):
            require(sha(main/name)==reference[key],'Main reference bytes differ')
        main_report=audit_session(main,freeze_path,_reference_depth=1)
        require(reference['gain']==main_report['selected_gain'],'Reference gain differs from independently replayed main fit')
    records=durable_records(session)
    replay=Replay(records,fixture,manifest['source'],reference=reference)
    try:
        actual=replay.run(protocol)
    finally:
        records.close()
    observations=json.loads((session/'observations.json').read_text(encoding='utf-8'))
    same(observations,actual,'observations semantic replay')
    for key in ('status','impacts','pulses','decisions','measured_images','gain'):
        same(manifest[key],actual[key],'final manifest.'+key)
    input_total=[sum(counts[bl] for counts in replay.input_events.values()) for bl in range(32)]
    diagnostics=actual['diagnostics']
    report=dict(schema='mnist-conductance-independent-audit-v1',status='PASS',
        session=session.relative_to(ROOT).as_posix(),source=manifest['source'],outcome=actual['status'],
        analyzer_sha256=sha(__file__),code_freeze_sha256=sha(freeze_path),protocol_sha256=digest(protocol),
        manifest_sha256=sha(session/'manifest.json'),raw_index_sha256=sha(session/'raw/index.json'),
        fixture_sha256=FIXTURE_SHA,selected_gain=actual['gain'],records=replay.records,
        native_impacts=replay.calls,pulses=replay.pulses,decisions=replay.decisions,measured_images=replay.ordinal,
        checkpoint_cells=256,learning_eligible_cells=128,control_cells=128,
        checkpoint_count=len(replay.checkpoints),mode_counts=dict(replay.mode_counts),
        maximum_predicted_mvm_current_a=replay.max_mvm_current,input_events_by_bl=input_total,
        no_image_event_bl=[bl for bl,n in enumerate(input_total) if n==0],
        image_cell_participation=replay.participation,diagnostics=diagnostics,
        native_no_update_reference=reference is not None,main_independent_audit=main_report,
        required_comparison_status='reference_and_main_replayed' if reference is not None else 'separate_no_update_reference_pending',
        hardware_launch_qualified=False,full10k_qualified=False,classification_accuracy_measured=False,
        model_or_connector_opened_by_auditor=False,
        limitations=['Raw/algorithm integrity is not a learning-benefit claim.',
            'Each conductance checkpoint is time-local; no recovery of physical state from digital checkpoints.',
            'Same native SIM initial seed does not preserve aligned stochastic draws after update paths diverge.',
            'Hardware/private-source transfer remains strict local source verification; no receipt fallback in this auditor.',
            'Hardware/full10k require a separate scientific decision including the planned no-update comparison.'])
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',required=True)
    parser.add_argument('--source-freeze',default='reports/2026-10-19/mnist-physical-design/PILOT128_CODE_FREEZE.json')
    parser.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    output=local(args.output)
    require(not output.exists(),'Choose a new audit output; evidence is immutable')
    report=audit_session(local(args.session),local(args.source_freeze))
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({key:report[key] for key in ('status','source','outcome','native_impacts','pulses','selected_gain','required_comparison_status')}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
