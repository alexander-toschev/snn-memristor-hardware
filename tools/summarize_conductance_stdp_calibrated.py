"""Summarize saved native-SIM evidence; no model/Connector or acquisition imports."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/2026-10-19/conductance-stdp-calibrated-review'
SESSION = ROOT / 'results/conductance_stdp_calibrated_20261019/d27da87e-9928-4303-971e-215afbc63158'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    audit = json.loads((REPORT / 'NATIVE_SIM_AUDIT.json').read_text())
    manifest = json.loads((SESSION / 'manifest.json').read_text())
    if audit['status'] != 'pass' or sha(SESSION / 'manifest.json') != audit['manifest_sha256']:
        raise ValueError('Expected audited native SIM session')
    if manifest['source'] != 'simulator' or manifest['status'] != 'completed':
        raise ValueError('Expected completed native SIM')
    for name, pin in manifest['result_sha256'].items():
        if sha(SESSION / name) != pin:
            raise ValueError('Saved result changed: ' + name)
    obs = json.loads((SESSION / 'observations.json').read_text())
    cells = []
    pulses = [p for p in obs['updates'] if p['phase'] == 'training_update']
    for wl in (5, 7):
        for bl in (6, 8, 9, 14):
            points = [p for p in obs['checkpoints'] if (p['wl'], p['bl']) == (wl, bl)]
            trajectory = [dict(phase=p['phase'], window=p.get('window_index'),
                               q=mean(p['q']), native_g_us=mean(p['native_inverse_r'])*1e6,
                               mvm_v=mean(r['centered_v'] for r in p['responses'])) for p in points]
            stages = {p['phase']: p for p in trajectory if p['phase'] in
                      ('initial', 'after_preparation', 'baseline_before_training', 'after_posttest', 'after_retention')}
            selected = [p for p in pulses if (p['wl'], p['bl']) == (wl, bl)]
            updates = [dict(window=p['window_index'], sign=p['sign'], action='SET' if p['sign'] else 'RESET',
                            eligibility=p['eligibility'], pulse_task_id=p['pulse_task_id'],
                            delta_q=330/p['pulse_native_r']-330/p['guard']['r'],
                            delta_native_g_us=(1/p['pulse_native_r']-1/p['guard']['r'])*1e6)
                       for p in selected]
            baseline, post, retention = (stages[s] for s in
                ('baseline_before_training', 'after_posttest', 'after_retention'))
            cells.append(dict(wl=wl, bl=bl, role='trained' if wl == 5 else 'control',
                              stages=stages, training_updates=updates,
                              training_checkpoint_trajectory=[p for p in trajectory if p['phase']=='after_training_window'],
                              delta_q_post=post['q']-baseline['q'],
                              delta_q_post_percent=100*(post['q']/baseline['q']-1),
                              delta_mvm_post_v=post['mvm_v']-baseline['mvm_v'],
                              delta_q_retention=retention['q']-post['q'],
                              delta_mvm_retention_v=retention['mvm_v']-post['mvm_v']))
    windows = []
    for w in obs['windows']:
        if w['phase'] == 'fit':
            continue
        banks = {}
        for wl in (5, 7):
            steps = [r for r in w['steps'] if r['wl'] == wl]
            banks[str(wl)] = dict(posts=[r['t'] for r in steps if r['post']],
                                 zero_input_posts=sum(r['post'] for r in steps if not any(r['pre'])),
                                 eligibility=w['eligibility'][str(wl)])
        windows.append(dict(phase=w['phase'], index=w['index'], banks=banks))
    pre = {w['index'] % 2: w['banks'] for w in windows if w['phase'] == 'pretest'}
    probes_match = all(all(w['banks'][wl]['posts'] == pre[w['index'] % 2][wl]['posts']
                          for wl in ('5','7')) for w in windows if w['phase'] in ('posttest','retention'))
    updates = [p for c in cells for p in c['training_updates']]
    sign_counts = Counter(p['action'] for p in updates)
    expected_direction = sum(p['delta_q'] > 0 if p['sign'] else p['delta_q'] < 0 for p in updates)
    summary = dict(schema='conductance-stdp-calibrated-native-sim-descriptive-v1',
                   session=SESSION.relative_to(ROOT).as_posix(), manifest_sha256=sha(SESSION/'manifest.json'),
                   audit_sha256=sha(REPORT/'NATIVE_SIM_AUDIT.json'), analysis_source_sha256=sha(Path(__file__)),
                   source='native MemriBoard simulator', physical_memristor=False,
                   selected_gain=obs['gain'], impacts=audit['impacts'], preparation_pulses=64,
                   training_pulses=len(updates), training_action_counts=dict(sign_counts),
                   pulse_direction_matches=expected_direction, pulse_direction_mismatches=len(updates)-expected_direction,
                   phase_spikes=audit['phase_spikes'], post_retention_probe_spike_times_match_pretest=probes_match,
                   zero_input_posts=sum(b['zero_input_posts'] for w in windows for b in w['banks'].values()),
                   control_q_and_mvm_unchanged=all(c['delta_q_post']==c['delta_mvm_post_v']==
                       c['delta_q_retention']==c['delta_mvm_retention_v']==0 for c in cells if c['wl']==7),
                   retention_elapsed_s=audit['retention_elapsed_s'], cells=cells, windows=windows,
                   hardware_approved=False, accuracy=None,
                   conclusion='Correct active logical-STDP pipeline in native SIM; no probe spike-pattern improvement demonstrated.',
                   limits=['Single seeded native-model session, eight cells, singleton input only.',
                           'q=330/nativeR is a dimensionless transfer proxy, not GUI weight or calibrated physical conductance.',
                           'Immediate per-pulse nativeR is available; MVM training effects are measured after each complete update window.',
                           'Control initial/prepared states differ; compare each bank against its own baseline, not raw spike totals.',
                           'Idle wall time does not evolve this simulator state; this is not physical retention evidence.',
                           'Logical pre/post timing and software eligibility do not establish intrinsic physical delta-t STDP.',
                           'NativeSIM100modelsteps is not a known physical board pulse duration.'])
    (REPORT/'NATIVE_SIM_SUMMARY.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    table=[]
    for c in cells:
        b,p,r=(c['stages'][s] for s in ('baseline_before_training','after_posttest','after_retention'))
        counts=Counter(u['action'] for u in c['training_updates'])
        table.append(f"| {c['wl']}/{c['bl']} | {counts['SET']}/{counts['RESET']} | {b['q']:.6f} → {p['q']:.6f} | {b['mvm_v']*1000:.2f} → {p['mvm_v']*1000:.2f} | {c['delta_q_post_percent']:+.2f}% | {r['q']:.6f} / {r['mvm_v']*1000:.2f} |")
    text = '''# STDP v2: фактический результат штатного симулятора MemriBoard

**Активная цепочка работает: появились post-спайки, eligibility обоих знаков
и32 импульса обучения. Улучшение выходного spike-pattern не показано: pretest,
posttest и retention дают одинаковые времена спайков для одинаковых проб.**
Это native SIM через публичный MemriBoard Connector, не эксперимент на плате
и не измерение точности CIFAR/MNIST.

Сессия `d27da87e-9928-4303-971e-215afbc63158`, clean tracked commit `daa2505292c85b95bb09b723df72b37ac2d2a895`.
Независимый [аудит](NATIVE_SIM_AUDIT.json) проверил provenance, raw, ADC,
выбор gain, LIF, eligibility, знаки, guards, контроль и полноту.
Собрано6616/6712 разрешённых impact:1840 mode7 и4776 mode10,
232 checkpoint,28 окон.64 импульса подготовки и32 обучения —96/128;
пропусков по току нет. Максимальный расчётный ток MVM8.50мкА.

## Калибровка и LIF

Из заранее заданной сетки gain[10,20,40,80,100,200,400,800,1600] допустимы800
и1600: оба дают по2 post-спайка в каждом fit-окне WL5. Выбран800В⁻¹ по
заранее заданному правилу минимального gain при равной цели. Gain заморожен
до pretest; WL7 и тестовые окна в подборе не использовались. Это цифровое
усиление перед LIF, не увеличение напряжения на ячейках: MVM остаётся0.05В.

В26 нефитовых окнах WL5 даёт52 спайка, WL7 —78: соответственно2 и3 на окно.
В шагах с нулевым входом спайков нет. Это не all-silent и не насыщение
«один выход на каждый из4 входов». Числа52 и78 нельзя сравнивать как выигрыш
обучения: подготовленные состояния двух банков различались ещё до обучения.

## Ячейки, запись и контроль

WL5 обучается, WL7 — контроль без обучающих импульсов. Оба банка получили
одинаковую последовательность подготовки; перед каждой попыткой обучения
добавлено парное контрольное guard-чтение. Внутреннее readback записи этим
не уравнивается. Сравниваем каждую ячейку с её собственным baseline после
подготовки, а не с исходным состоянием до64 подготовительных импульсов.

q=330/nativeR — безразмерный прокси передачи; это не GUI weight
`Rload/(Rload+R)` и не независимо калиброванная физическая проводимость.
MVM — средний центрированный отклик трёх свежих zero/signal/zero измерений.

| WL/BL | SET/RESET обучения | q до → после | MVM до → после, мВ | Δq | После60с: q / MVM, мВ |
|---|---:|---|---|---:|---|
''' + '\n'.join(table) + '''

BL6 иBL9 не исключались: их eligibility в train равна+0.135335 и−0.142102,
ниже порога|E|≥0.25, поэтому32 решения корректно дали no_update.
УBL8 E чередуется+0.596338/−0.928410, уBL14 +0.884200/−0.643967.
В каждом чётном окне обе ячейки получают SET1.7В/sign1, в нечётном —
RESET1.4В/sign0:16 SET и16 RESET суммарно, все приняты штатными guards.

BL8 после окон меняет q в диапазоне0.003290…0.018904, MVM0…0.92мВ,
но последний RESET возвращает q и MVM к baseline. BL14 изменяет q в диапазоне
0.003290…0.012918 и заканчивает на71.53% ниже baseline; отклик падает
с0.61мВ до0. Положительный nativeR остаётся: нулевой ADC-отклик не означает
нулевую физическую проводимость. Примерно один ADC LSB равен0.305мВ, поэтому
значительная часть наблюдаемой MVM-динамики находится на уровне1–3 отсчётов.

По непосредственному readback31 из32 импульсов имеют ожидаемый знак Δq:
15/16 SET повышают q,16/16 RESET понижают. Первый SET BL14 даёт
**Δq=−0.00131037** (около−9.21%): знак команды не гарантирует знак измеренного
изменения даже в этом штатном SIM. В исходниках модели есть шум состояния
при программировании; это возможное объяснение, а не отдельно доказанная
причина конкретного отклонения. Native readback и последующий checkpoint
могут немного различаться. Индивидуальный ΔMVM каждого обучающего импульса
не измерялся: MVM checkpoint следует после всех обновлений окна.

Все4 контрольные ячейки сохраняют q и MVM в posttest и retention; их
межоконные checkpoint также постоянны. У обучаемых BL6/9 без обновлений
изменений нет. Это поддерживает связь динамики BL8/14 с обновлениями в данной
модели, но один контрольный банк с иными начальными состояниями не даёт
обобщённой оценки причинного эффекта или статистической значимости.

## Что показали пробы после обучения

Для обеих чётностей heldout-проб posttest и retention повторяют pretest
**точно по временам спайков**: WL5 [1,9]; WL7 [4,9,12] или[1,4,12].
Входные проводимости менялись, но пороговый выход LIF на этих пробах — нет.
Обучение не имеет меток/целевого качества, поэтому нельзя объявить рост
точности, полезное распознавание или успешное изучение нового шаблона.

После60с паузы измеренные q/MVM не изменились. В этой модели состояние
обновляется вызовами apply_voltage; отдельной эволюции при ожидании wall-time
нет. Поэтому это проверка паузы/повторного чтения протокола, **не доказательство
физического удержания проводимости в течение60с**.

## Вывод и граница переноса

Подтверждены активный LIF, независимый fit-only gain, logical-STDP eligibility,
оба знака обновления, конечный бюджет и контроль без обучающих записей.
Не подтверждены полезность обучения по выходному шаблону, физическая STDP
от наложения pre/post-импульсов, аппаратная воспроизводимость или точность сети.
В SIM длительность —100 шагов модели; длительность штатного импульса платы
по-прежнему неизвестна. Guard до импульса ограничивает расчёт по свежему R,
а не измеряет внутрипульсный ток; внутренний signed-current clamp native SIM
имеет ранее отмеченное отличие и не исправлялся для получения результата.

Аппаратный запуск требует отдельного решения validator/регистрации и идёт
после CIFAR. Все реальные чтения/SET/RESET/MVM — только public MemriBoard
Connector, штатные converters и INI. Нового I/O в этом анализе не было.

[Машинная сводка с полными траекториями](NATIVE_SIM_SUMMARY.json).
Воспроизведение анализа: `python tools/summarize_conductance_stdp_calibrated.py`.
'''
    (REPORT/'REPORT_RU.md').write_text(text,encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('training_action_counts','pulse_direction_matches','post_retention_probe_spike_times_match_pretest','control_q_and_mvm_unchanged')}))


if __name__ == '__main__':
    main()
