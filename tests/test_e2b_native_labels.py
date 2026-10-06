"""Candidate label alignment without changing scoring or generic option semantics."""
import copy
import json
import unittest

from adaptive_policy import DecisionPipeline, keys_for, load_policy, native_option_lines, rendered_options
from test_adaptive_pipeline import FakeTransport
from test_e2b_release import POLICY, inspection, scored, completed
from reasoning_contract import load_profile


class E2BNativeLabels(unittest.TestCase):
    def test_native_names_descriptions_order_images_and_owner_state_survive(self):
        cases = [
            {'type':'choice','criteria':{'red':None,'blue':None}},
            {'type':'choice','criteria':{'z':{'v':'<tag>'},'a':[], 'empty':''}},
            {'type':'noul','criteria':{'true':'yes','false':'no'}},
            {'type':'noul'},
            {'type':'score','criteria':[None,{'rating':'high'},['top']]},
            {'type':'choice','criteria':{f'candidate-{i}':None for i in range(64)}},
        ]
        all_labels = [chr(65+i) for i in range(26)] + ['A'+chr(65+i) for i in range(26)] + ['B'+chr(65+i) for i in range(12)]
        # Nonalphabetical inspection mapping proves the implementation does not infer labels.
        all_labels[0],all_labels[1] = all_labels[1],all_labels[0]
        for question in cases:
            for images in (False,True):
                with self.subTest(question=question, images=images):
                    q = dict(question,instructions='Which candidate matches the evidence?')
                    request = {'state':{'model_reasoning':'owner data','facts':['<marker>']},'questions':{'q':q}}
                    if images:
                        request['winnow']={'images':['data:image/png;base64,YQ==','data:image/png;base64,Yg==']}
                    saved = copy.deepcopy(request)
                    profile=load_profile('e2b-q8-'+('vision' if images else 'text')+'8k-mtp')
                    inspected=inspection(profile,images)
                    inspected['runtime'].update(labels=all_labels,label_token_ids=list(range(100,164)))
                    manifest,_=load_policy(POLICY)
                    manifest['policies'][POLICY]['generation']['prompt_format']='e2b-native-labels-v3'
                    keys=keys_for(q);logits=[0]*len(keys)
                    transport=FakeTransport([inspected,scored(request,logits,'on'),completed(),inspected,scored(request,logits,'on')])
                    result=DecisionPipeline(transport,reasoning='always',policy_id=POLICY,policy_manifest=manifest,runtime_profile=profile).decide(request)
                    self.assertTrue(result['winnow']['adaptive']['completed_blend'])
                    parts=transport.calls[2][1]['messages'][0]['content']
                    text=parts[-1]['text'] if images else parts
                    expected=native_option_lines(rendered_options(q,keys),inspected['runtime']).replace('<','\\u003c')
                    self.assertEqual(text.split('\nOptions:\n')[1],expected)
                    self.assertEqual(transport.calls[1][1]['questions'],request['questions'])
                    self.assertEqual(transport.calls[4][1]['questions'],request['questions'])
                    self.assertEqual(transport.calls[4][1]['state']['original_state'],saved['state'])
                    self.assertEqual(request,saved)
                    if images:
                        self.assertEqual([p['image_url']['url'] for p in parts if p['type']=='image_url'],request['winnow']['images'])

    def test_invalid_or_missing_inspected_mapping_is_rejected(self):
        valid={'labels':['A','C'],'label_token_ids':[10,12]}
        for invalid in ({}, {'labels':['A','A'],'label_token_ids':[10,12]},
                        {'labels':['A','C'],'label_token_ids':[10,10]},
                        {'labels':['A','C'],'label_token_ids':[True,12]},
                        {'labels':['A','C'],'label_token_ids':[10]},
                        {'labels':['1','2'],'label_token_ids':[10,12]},
                        {'labels':['A','AAA'],'label_token_ids':[10,12]}):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):
                native_option_lines(['red','blue'],invalid)
        self.assertEqual(native_option_lines(['red','blue'],valid),'A: "red"\nC: "blue"')

    def test_format_is_versioned_and_v2_is_retained_without_policy_tuning(self):
        import assets
        old,old_policy=load_policy('e2b-raw99-blend50-v2')
        current,current_policy=load_policy(POLICY)
        self.assertEqual(old['generation']['prompt_format'],'e2b-canonical-v2')
        self.assertEqual(current['generation']['prompt_format'],'e2b-native-labels-v3')
        self.assertEqual(old_policy['policy'],current_policy['policy'])
        for field in ('target','assistant','profile','runtime_contract_required'):
            self.assertEqual(old_policy[field],current_policy[field])
        self.assertEqual(assets.selection('e2b')[0]['policy'],POLICY)

if __name__ == '__main__':
    unittest.main()
