"""Offline future-HW replay and descriptive comparison; writes new reports only."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'.deps')]
from tools.audit_mnist_diagonal import audit
from experiments.unn_rev3_20261014.paths import repo_path


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def summarize(session,replayed):
    o=read(session/'observations.json');m=read(session/'manifest.json');phase={}
    for name,v in o['diagnostics']['phases'].items():
        posts=v['post_counts_by_wl'];n=v['images']
        phase[name]=dict(images=n,posts_by_wl=[sum(x) for x in posts],
            learning_posts=sum(sum(x) for x in posts[:4]),control_posts=sum(sum(x) for x in posts[4:]),
            learning_posts_per_image=sum(sum(x) for x in posts[:4])/n,
            control_posts_per_image=sum(sum(x) for x in posts[4:])/n,
            silent_images_by_wl=v['silent_images_by_wl'],unique_eight_neuron_outputs=v['unique_eight_neuron_outputs'],
            all_learning_traces_identical=v['all_learning_traces_identical'])
    train=[u for u in o['updates'] if u['phase']=='training_update'];writes=[u for u in train if u['admitted']]
    bysign={}
    for sign,name in [(0,'RESET'),(1,'SET')]:
        rows=[u for u in writes if u['sign']==sign];ds=[u['delta_inverse_r'] for u in rows]
        bysign[name]=dict(admitted=len(rows),expected_delta_g=sum((x>0 if sign else x<0) for x in ds),
            opposite_delta_g=sum((x<0 if sign else x>0) for x in ds),exact_zero_delta_g=sum(x==0 for x in ds),delta_g_sum_siemens=sum(ds))
    cp={r['phase']:r['mean_inverse_r'] for r in o['checkpoints']}
    def delta(before,after):
        if before not in cp or after not in cp:return None
        banks={}
        for name,wls in [('learning',range(4)),('control',range(4,8))]:
            ds=[cp[after][w][b]-cp[before][w][b] for w in wls for b in range(32)]
            banks[name]=dict(nonzero_cells=sum(d!=0 for d in ds),mean_delta_siemens=sum(ds)/len(ds),max_abs_delta_siemens=max(map(abs,ds)))
        return dict(start=before,end=after,banks=banks)
    return dict(schema='mnist-diagonal-hardware-summary-v1',source=m['source'],session=session.relative_to(ROOT).as_posix(),
        status=m['status'],independent_replay=replayed['status'],manifest_sha256=sha(session/'manifest.json'),
        observations_sha256=sha(session/'observations.json'),protocol_sha256=m['protocol_sha256'],
        gain=o['gain'],per_wl_selected=o['calibration'].get('per_wl_selected'),
        shared_gain_diagnostic=o['calibration'].get('shared_gain_diagnostic'),
        impacts=m['impacts'],pulses=m['pulses'],preparation_pulses=sum(u['admitted'] for u in o['updates'] if u['phase']=='preparation'),
        measured_presentations=o['measured_images'],training_presentations=sum(x['phase']=='train' for x in o['outcomes']),
        elapsed_seconds=(datetime.fromisoformat(m['finished_utc'])-datetime.fromisoformat(m['started_utc'])).total_seconds(),phase=phase,
        training=dict(attempts=len(train),admitted=len(writes),guard_skipped=sum(not u['guard_admissible'] for u in train),
            by_sign=bysign,unique_updated_cells=len({(u['wl'],u['bl']) for u in writes}),
            updated_cells_by_wl=[len({u['bl'] for u in writes if u['wl']==w}) for w in range(4)]),
        coverage=dict(checkpoint_cells=256,learning_eligible_cells=128,
            image_contributing_cells=sum(x>0 for row in o['image_cell_participation'] for x in row),
            no_image_event_bl=[b for b in range(32) if not any(o['input_event_counts'][phase][b] for phase in o['input_event_counts'])]),
        training_g_change=delta('after_preprobe','after_train_128'),retention_g_change=delta('after_postprobe','after_retentionprobe'),
        classification_accuracy_measured=False,full10k_executed=False,
        private_runtime_status='private_runtime_not_independently_reverified')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',required=True);p.add_argument('--output-dir',required=True)
    args=p.parse_args();session=repo_path(args.session);out=repo_path(args.output_dir)
    if read(session/'manifest.json')['source']!='hardware':raise ValueError('This report is for new HW only')
    names=('AUDIT.json','SUMMARY.json','REPORT_RU.md','SIM_COMPARISON.json')
    if any((out/n).exists() for n in names):raise FileExistsError('Keep existing reports immutable')
    result=audit(session);summary=summarize(session,result)
    sim_path=ROOT/'reports/2026-10-20/mnist-diagonal/SUMMARY.json';sim=read(sim_path)
    sim_manifest=read(ROOT/sim['session']/'manifest.json')
    if sim_manifest['protocol_sha256']!=summary['protocol_sha256']:raise ValueError('Different protocol; cannot use same-protocol comparison')
    comparison=dict(schema='mnist-diagonal-hw-vs-sim-descriptive-v1',sim_summary_sha256=sha(sim_path),
        simulator_session=sim['session'],hardware_session=summary['session'],gains=dict(hardware=summary['gain'],simulator=sim['gain']),
        phase={},training=dict(hardware=summary['training'],simulator=sim['training']),
        coverage=dict(hardware=summary['coverage'],simulator=sim['coverage']),
        retention_g_change=dict(hardware=summary['retention_g_change'],simulator=sim['retention_g_change']),
        limitations=['Each fresh fit selects its own diagonal; this is not an identical fixed-gain intervention.',
                    'Same fixture/protocol does not make SIM and HW initial conductances identical.',
                    'Post-count or delta-G sign differences do not establish accuracy or learning benefit.'])
    for name in ('preprobe','train','postprobe','retentionprobe'):
        comparison['phase'][name]=dict(hardware=summary['phase'].get(name),simulator=sim['phase'].get(name))
    lines=['# Аппаратный диагональный MNIST128','',f"Сеанс `{session.name}`: `{summary['status']}`; независимый raw replay **PASS**.",
        f"Gain WL0–7: `{summary['gain']}`. {summary['impacts']} воздействий/{summary['pulses']} импульсов; training images={summary['training_presentations']}.",'']
    if summary['status']=='uninformative_calibration':
        lines+=['Хотя бы один WL не получил допустимый индивидуальный gain. Train/probes не выполнены; это не нулевая accuracy и не доказательство невозможности обучения.','']
    lines+=['|Фаза|Изображений HW|Спайки HW learning/control|Спайки SIM learning/control|','|---|---:|---:|---:|']
    for name,v in comparison['phase'].items():
        hw=v['hardware'];sv=v['simulator']
        lines.append(f"|{name}|{hw['images'] if hw else 'не измерено'}|{str(hw['learning_posts'])+'/'+str(hw['control_posts']) if hw else '—'}|{str(sv['learning_posts'])+'/'+str(sv['control_posts']) if sv else '—'}|")
    lines+=['','|Запись HW|Импульсов|Ожидаемый ΔG|Противоположный|Точно нулевой|','|---|---:|---:|---:|---:|']
    for name,r in summary['training']['by_sign'].items():lines.append(f"|{name}|{r['admitted']}|{r['expected_delta_g']}|{r['opposite_delta_g']}|{r['exact_zero_delta_g']}|")
    lines+=['',f"Training обновлял {summary['training']['unique_updated_cells']} разных адресов; eligible128. Image-contributing: {summary['coverage']['image_contributing_cells']}/256; без input events BL{summary['coverage']['no_image_event_bl']}. Checkpoints/preparation coverage не равно обучению всех256."]
    r=summary['retention_g_change']
    if r:lines.append(f"Retention post→retention: ненулевой ΔG learning/control {r['banks']['learning']['nonzero_cells']}/{r['banks']['control']['nonzero_cells']}; max|ΔG|={r['banks']['learning']['max_abs_delta_siemens']:.6g}/{r['banks']['control']['max_abs_delta_siemens']:.6g}S.")
    lines+=['','Метрики learning/control разделены; gain выбран только freshfit. Точно нулевой ΔG определяется точным равенством, знак ожидается по рецепту SET/RESET; это не оценка полезности обучения. Диагонали HW/SIM могут различаться. Уменьшение/рост числа спайков не является accuracy. Этот опыт не содержит classifier/readout и не full10k.','']
    out.mkdir(parents=True,exist_ok=True)
    save(out/'AUDIT.json',result);save(out/'SUMMARY.json',summary);save(out/'SIM_COMPARISON.json',comparison)
    with (out/'REPORT_RU.md').open('x',encoding='utf-8',newline='\n') as f:f.write('\n'.join(lines))
    print(json.dumps(dict(status=summary['status'],audit=result['status'],gain=summary['gain'],training_images=summary['training_presentations'],training_pulses=summary['training']['admitted'],output_dir=args.output_dir)))

if __name__=='__main__':main()
