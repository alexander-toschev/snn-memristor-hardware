"""Offline hardware-branch fakes and qualification boundaries; no native device."""
import copy
import unittest
from unittest import mock
from experiments.conductance_stdp_hardware_20261019 import experiment as e
from experiments.conductance_stdp_calibrated_20261019 import core
from tests.test_conductance_stdp_calibrated import VariedConnector
from tests.test_conductance_stdp import Trace,Clock,Converters
from tools import audit_conductance_stdp_hardware as audit


class FakeHardware(VariedConnector):cb_type='real'


class HardwareTests(unittest.TestCase):
    def test_closed_gate_precedes_any_preflight_or_connector(self):
        with mock.patch.object(e,'APPROVED_LAUNCH_SPEC_SHA256',frozenset()),mock.patch.object(e,'preflight') as preflight:
            with self.assertRaisesRegex(ValueError,'review and registration'):e.main(['--run-hardware'])
            preflight.assert_not_called()

    def test_actual_qualified_sim_and_frozen_source_inventory(self):
        freeze,qualified=e.qualification()
        self.assertEqual(freeze['protocol_sha256'],e.PROTOCOL_SHA)
        self.assertTrue(qualified['sim_gain_not_reused'])
        self.assertEqual(qualified['reference_sha256'][str((e.REVIEW/'NATIVE_SIM_INDEPENDENT_AUDIT.json').relative_to(e.ROOT)).replace('\\','/')],e.SIM_AUDIT_SHA)

    def test_fake_hardware_native_width_and_fresh_gain(self):
        connector=FakeHardware();trace=Trace();clock=Clock()
        runner=core.Collector(connector,Converters,trace,'hardware',clock.sleep,clock.now)
        with mock.patch('builtins.print'):result=runner.run()
        self.assertEqual(result['status'],'completed');self.assertEqual(result['gain'],400.)
        writes=[c for c in connector.calls if c['mode_flag']==7 and c['vol']]
        self.assertTrue(writes);self.assertTrue(all(c['pulse_width']==0. for c in writes))
        self.assertTrue(all(row['source']=='hardware' for row in trace.rows))
        self.assertLessEqual(result['impacts'],6712);self.assertLessEqual(result['pulses'],128)
        self.assertTrue(connector.closed and trace.finished)
        self.assertTrue(all(u['wl']==5 for u in result['updates'] if u['phase']=='training_update'))

    def test_supplemental_start_final_qualification_tampering(self):
        fields=('session_id','source','source_sha256','protocol_sha256','launcher_sha256',
                'launcher_source_sha256','launcher_spec','launcher_spec_sha256','qualification',
                'qualifying_sim_audit_sha256','invocation','physical_pulse_duration_s','seed','environment')
        start={key:{'value':key} for key in fields};finish=copy.deepcopy(start)
        audit.compare_start_final(start,finish)
        for key in ('qualification','launcher_sha256','launcher_source_sha256','source_sha256','physical_pulse_duration_s'):
            changed=copy.deepcopy(finish);changed[key]={'value':'tampered'}
            with self.assertRaisesRegex(ValueError,'mismatch'):audit.compare_start_final(start,changed)

    def test_no_supplemental_audit_without_registered_launcher(self):
        with mock.patch.object(audit,'APPROVED_LAUNCHER_SHA256',frozenset()):
            with self.assertRaisesRegex(ValueError,'independently registered'):audit.launcher_attestation(e.SIM)


if __name__=='__main__':unittest.main()
