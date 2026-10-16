"""Synthetic native-shaped request/result mutations; no native backend."""
import copy
import unittest
import native_full_cifar_link as n


def specimen(active=True,explicit_xy=True):
    times=[None]*3072
    if active:times[13*32+13]=0
    images=[dict(case=dict(original_test_index=9,image_chw_uint8_sha256="a"*64),times=times)]
    slots=[dict(tap_index=0,positive_cell=[0,1],negative_cell=[1,2])]
    parser=n.NativeLink("synthetic","synthetic_test",0,slots,images,[[1.,.25],[.5,2.]],1.,"r","c","m",
                        historical_xy=None if explicit_xy else (13,13))
    events=[dict(event="route_frozen",session_id="synthetic",sha256="r",before_task_id=1),
            dict(event="calibration_frozen",session_id="synthetic",sha256="c",matrix_sha256="m",before_task_id=1)]
    for i,(side,wl,bl,role,adc) in enumerate([
        ("positive_cell",0,1,"zero_before",100),
        ("positive_cell",0,1,"signal" if active else "empty_zero_signal",400),
        ("positive_cell",0,1,"zero_after",200),
        ("negative_cell",1,2,"zero_before",50),
        ("negative_cell",1,2,"signal" if active else "empty_zero_signal",150),
        ("negative_cell",1,2,"zero_after",100)],1):
        vector=[0.]*32
        if role=="signal":vector[bl]=.1
        req=dict(event="request",session_id="synthetic",source="synthetic_test",task_id=i,phase="cifar",
                 filter_id=0,case=0,original_test_index=9,repeat=0,t=0,side=side,role=role,wl=wl,bl=bl,
                 command_si=dict(mode_flag=10,id=0,wl=wl,vol=vector),native_adc_samples=1,
                 native_voltage_conversion=[dict(requested_v=v,dac_code=82 if v else 0,native_decoded_v=.1 if v else 0.) for v in vector])
        if explicit_xy:req.update(output_y=13,output_x=13)
        if role=="signal":req["tap"]=0
        result={**copy.deepcopy(req),"event":"result","status":"ok","native_result":[123.,0,adc],
                "returned_id":0,"raw_adc":adc,"output_voltage_v":n.volts(adc)}
        events.extend((req,result))
    return parser,events


def replay(parser,events):
    out=[]
    for event in events:out.extend(parser.accept(event))
    out.extend(parser.finish())
    return out


class NativeLinkTests(unittest.TestCase):
    def test_exact_raw_refs_adc_and_hand_calculation(self):
        parser,events=specimen();out=replay(parser,events)
        self.assertEqual(len(out),1)
        v=out[0]
        self.assertEqual(v["terms"][0]["raw_task_ids"],[1,2,3,4,5,6])
        self.assertEqual(v["terms"][0]["positive_adc"],[100,400,200])
        self.assertEqual(v["active_taps"],[0])
        self.assertAlmostEqual(v["trained_unit_contribution"],.5*.07630-1.75*.02289,places=12)
        self.assertEqual(parser.pairs,6)

    def test_fresh_empty_zero_is_preserved_not_cpu_zero(self):
        parser,events=specimen(active=False);out=replay(parser,events)
        self.assertEqual(out[0]["active_taps"],[])
        self.assertIsNone(out[0]["terms"][0]["signal_tap"])
        self.assertNotEqual(out[0]["trained_unit_contribution"],0.)

    def test_raw_adc_and_native_result_cannot_diverge(self):
        parser,events=specimen();events[3]["raw_adc"]+=1
        with self.assertRaises(ValueError):replay(parser,events)

    def test_wrong_dac_or_returned_id_rejected(self):
        for field in ("dac","id","native_voltage"):
            parser,events=specimen()
            if field=="dac":
                for i in (4,5):events[i]["native_voltage_conversion"][1]["dac_code"]=81
            elif field=="id":events[3]["returned_id"]=7
            else:events[3]["output_voltage_v"]+=.001
            with self.assertRaises(ValueError):replay(parser,events)

    def test_wrong_TTFS_tap_BL_or_image_rejected(self):
        for field in ("bl","tap","image"):
            parser,events=specimen()
            for i in (4,5):
                if field=="tap":events[i]["tap"]=1
                elif field=="image":events[i]["original_test_index"]=132
                else:
                    events[i]["command_si"]["vol"][1]=0.
                    events[i]["command_si"]["vol"][3]=.1
                    events[i]["native_voltage_conversion"]=[dict(requested_v=v,dac_code=82 if v else 0,native_decoded_v=.1 if v else 0.) for v in events[i]["command_si"]["vol"]]
            with self.assertRaises(ValueError):replay(parser,events)

    def test_full_adapter_requires_explicit_spatial_metadata(self):
        parser,events=specimen()
        for row in events:
            row.pop("output_y",None);row.pop("output_x",None)
        with self.assertRaises(ValueError):replay(parser,events)

    def test_missing_duplicate_or_orphan_request_result_rejected(self):
        for mutation in ("missing","duplicate","orphan"):
            parser,events=specimen()
            if mutation=="missing":events.pop()
            elif mutation=="duplicate":events.insert(4,copy.deepcopy(events[2]))
            else:events.pop(2)
            with self.assertRaises(ValueError):replay(parser,events)

    def test_changed_matrix_freeze_or_late_freeze_rejected(self):
        parser,events=specimen();events[1]["matrix_sha256"]="changed"
        with self.assertRaises(ValueError):replay(parser,events)
        parser,events=specimen();events.pop(1)
        with self.assertRaises(ValueError):replay(parser,events)

    def test_repeated_native_measurements_cannot_fill_new_time(self):
        parser,events=specimen();extra=copy.deepcopy(events[2:])
        for row in extra:row["task_id"]+=6
        # Same case/repeat/time/location is a duplicate even with fresh task IDs.
        with self.assertRaises(ValueError):replay(parser,events+extra)


if __name__=="__main__":unittest.main()
