import tempfile
import unittest
from pathlib import Path
from cfra.export_handoff import write_export_handoff
from cfra.export_acceptance import write_acceptance
from collections import defaultdict


class ExportHandoffTests(unittest.TestCase):
    def test_handoff_requires_individual_stage_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'manifest').mkdir()
            for name in ['run.json','models.csv','images.csv','objects.csv','splits.csv']:
                (root/'manifest'/name).write_text('{}')
            config={'scope':{'experiment_ids':['E04-2'],'models':['mae'],
                             'datasets':['fair1m'],'perturbations':['blur']},'split':{'seed':1729}}
            result=write_export_handoff(root,config)
            self.assertEqual(result['status'],'incomplete')
            self.assertEqual(result['missing'][0]['stage'],'E04-2')
            folder=root/'intermediates/stage_results'; folder.mkdir(parents=True)
            (folder/'E04-2_mae.json').write_text('{"results": []}')
            result=write_export_handoff(root,config)
            self.assertEqual(result['status'],'ready_for_export_validation')
            self.assertEqual(result['scientific_validation'],'not_performed_by_handoff')

    def test_empty_export_never_claims_completeness(self):
        with tempfile.TemporaryDirectory() as tmp:
            report=write_acceptance(Path(tmp),defaultdict(list),[])
            self.assertFalse(report['export_ready'])
            self.assertTrue(report['gaps'])
