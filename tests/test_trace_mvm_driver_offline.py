from pathlib import Path
import tempfile
import unittest
from tools.trace_mvm_driver_offline import trace

SYNTHETIC = '''
raise RuntimeError("MODULE MUST NOT EXECUTE")
class Driver:
    def __init__(self):
        self.mvm_css = [13,19,26,21]
        raise RuntimeError("INIT MUST NOT EXECUTE")
    def mode_mvm(self, volts, tms, tus, rtms, rtus, wl, task_id):
        self.fpga.BLINK_ON()
        bytes_r = FPGA_SIM.gen_registers_bytes(wl,0)
        route = [int(bytes_r[0][2:][::-1],2),int(bytes_r[1][2:][::-1],2)]
        self.fpga.spi_send(route,99,1)
        volts = list(map(int, np.where(np.array(volts)>246,246,volts)))
        for i in range(4):
            for j in range(8):
                value = volts[i*8+j]
                self.fpga.spi_send([j<<4 | value>>8,value&255],self.mvm_css[i],1)
        self.fpga.ldac_mvm()
        self.fpga.spi_read(0,2)
        self.fpga.BLINK_OFF()
        return (0,task_id)
'''


class TraceTests(unittest.TestCase):
    def test_only_method_executes_and_all_four_chip_bytes_preserved(self):
        with tempfile.TemporaryDirectory(dir='.tmp') as directory:
            path=Path(directory)/'private_synthetic.py'
            path.write_text(SYNTHETIC)
            result=trace(path)
        self.assertFalse(result['hardware_io'])
        self.assertFalse(result['module_imported'])
        self.assertTrue(result['routing_stubbed'])
        for case in result['cases']:
            sent=[e for e in case['events'] if e['operation']=='spi_send' and e['chip']!=99]
            routing=[e for e in case['events'] if e['operation']=='spi_send' and e['chip']==99]
            self.assertEqual(routing[0]['values'],[0,0])
            self.assertEqual(case['events'][0]['operation'],'BLINK_ON')
            self.assertEqual(case['events'][-1]['operation'],'BLINK_OFF')
            self.assertEqual(len(sent),32)
            for b,event in enumerate(sent):
                self.assertEqual(event['chip'],[13,19,26,21][b//8])
                self.assertEqual(event['values'],[(b%8)<<4,case['input_dac'][b]])

    def test_method_import_rejected_without_execution(self):
        source=SYNTHETIC.replace('volts = list(map', 'import os\n        volts = list(map')
        with tempfile.TemporaryDirectory(dir='.tmp') as directory:
            path=Path(directory)/'private_synthetic.py'
            path.write_text(source)
            with self.assertRaises(ValueError):trace(path)


if __name__=='__main__':unittest.main()
