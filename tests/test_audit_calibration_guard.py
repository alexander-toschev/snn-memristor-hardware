"""Offline, manufactured journal events only. No runtime imports or Connector."""
import unittest

from audit_calibration_generic import GuardReplay,valid_invocation


class GuardAuditTest(unittest.TestCase):
    def test_cli_flag_order_allowed_but_extra_or_mixed_modes_rejected(self):
        for source in ('hardware','simulator'):
            flags=['--run-'+source,'--calibrate-cells']
            for pair in (flags,flags[::-1]):self.assertTrue(valid_invocation(['experiment.py']+pair,source))
        for argv in (['experiment.py','--run-hardware','--run-simulator'],
                     ['experiment.py','--calibrate-cells','--run-simulator','--run-hardware'],
                     ['cell_calibration.py','--calibrate-cells','--run-simulator']):
            self.assertFalse(valid_invocation(argv,'simulator'))

    def setUp(self):
        self.errors=[]
        def check(ok,label):
            if not ok:self.errors.append(label)
        def compare(a,b,label):check(a==b,label)
        self.guard=GuardReplay(check,compare,.002,'hardware')

    def read(self,r=1500,tid=5,wl=0,bl=1,mode=7,status='ok'):
        self.guard.observe(dict(status=status,command_si=dict(mode_flag=mode),
            native_r=r,task_id=tid,wl=wl,bl=bl))

    def event(self,skip=False,r=1500,rid=5,did=1,step=270,sign=1,before=35,wl=0,bl=1):
        v=round(.3+.01*step,2);limit=.002 if sign else .04
        task=dict(mode_flag=7,vol=v,pulse_width=0,sign=sign,wl=wl,bl=bl)
        if not skip:task['id']=before+1
        event=dict(event='skipped' if skip else 'request',wl=wl,bl=bl,phase='set' if sign else 'reset',step=step,
            programming_decision_id=did,previous_native_r_ohm=r,resistance_source_task_id=rid,
            predicted_current_a=v/r,current_limit_a=limit)
        if skip:event.update(planned_command_si=task,reason='predicted_write_current_exceeds_limit')
        else:event['command_si']=task
        return event

    def test_both_sign_boundaries_allow_equality(self):
        for sign,r in ((1,1500),(0,75)):
            with self.subTest(sign=sign):
                self.setUp();self.read(r)
                self.guard.decide(self.event(r=r,sign=sign),35)
                self.assertEqual(self.errors,[])

    def test_both_sign_above_boundary_requires_skip(self):
        for sign,r in ((1,1499),(0,74)):
            with self.subTest(sign=sign):
                self.setUp();self.read(r)
                self.guard.decide(self.event(skip=True,r=r,sign=sign),35)
                self.assertEqual(self.errors,[])
                self.assertEqual(self.guard.skipped,1)

    def test_current_pass_must_not_be_skipped(self):
        self.read();self.guard.decide(self.event(skip=True),35)
        self.assertIn('guard/1/allow_or_skip',self.errors)

    def test_current_fail_must_not_be_submitted(self):
        self.read(1499);self.guard.decide(self.event(r=1499),35)
        self.assertIn('guard/1/allow_or_skip',self.errors)

    def test_latest_valid_mode7_not_mode10_or_failed_mode7(self):
        self.read(1500,5);self.read(1600,7)
        self.read(123,8,mode=10);self.read(234,9,status='error')
        self.guard.decide(self.event(r=1600,rid=7),35)
        self.assertEqual(self.errors,[])

    def test_stale_readback_and_wrong_cell_are_detected(self):
        self.read(1500,5);self.read(1600,7)
        self.guard.decide(self.event(),35)
        self.assertIn('guard/1/resistance_source_task_id',self.errors)
        with self.assertRaises(ValueError):self.guard.decide(self.event(bl=2,did=2),36)

    def test_skip_does_not_update_r_or_consume_impact_id(self):
        self.read(100,5)
        for i in range(3):
            self.guard.decide(self.event(skip=True,r=100,rid=5,did=i+1,step=i),35)
        self.assertEqual(self.errors,[])
        self.assertEqual(self.guard.latest,{(0,1):(100,5)})
        self.assertEqual([d['before_record_count'] for d in self.guard.decisions],[35]*3)
        self.assertEqual([d['step'] for d in self.guard.decisions],[0,1,2])

    def test_fabricated_skip_measurements_and_command_id_rejected(self):
        for field,value in (('task_id',36),('raw_adc',20),('q',.01),('native_result',[100,36,20])):
            with self.subTest(field=field):
                self.setUp();self.read(100)
                event=self.event(skip=True,r=100,step=0);event[field]=value
                self.guard.decide(event,35)
                self.assertIn('guard/1/no_fabricated_measurement',self.errors)
        self.setUp();self.read(100)
        event=self.event(skip=True,r=100,step=0);event['planned_command_si']['id']=36
        self.guard.decide(event,35)
        self.assertIn('guard/1/exact_command',self.errors)

    def test_wrong_decision_id_voltage_or_native_current_is_detected(self):
        self.read()
        event=self.event();event.update(programming_decision_id=2,predicted_current_a=.0001)
        event['command_si']['vol']=2.9
        self.guard.decide(event,35)
        for suffix in ('programming_decision_id','predicted_current_a','ramp'):
            self.assertIn('guard/1/'+suffix,self.errors)

    def test_accepted_write_then_checkpoint_updates_next_guard(self):
        self.read(1500,5);self.guard.decide(self.event(step=0),35)
        self.read(1400,36);self.read(1300,41)
        self.guard.decide(self.event(r=1300,rid=41,did=2,step=1,before=71),71)
        self.assertEqual(self.errors,[])

    def test_simulator_duration_is_125_model_steps_only(self):
        self.guard.source='simulator';self.read()
        event=self.event();event['command_si']['pulse_width']=.000125
        self.guard.decide(event,35)
        self.assertEqual(self.errors,[])


if __name__=='__main__':unittest.main()
