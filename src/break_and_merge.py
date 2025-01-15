import numpy as np
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn import metrics
from pathlib import Path
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import src.utils as utils
from src.factscore_utils import FactScorer, General_Wiki_Eval

def convert_to_claim_list(breakdown_dicts):
    return [dict_['claim'] for dict_ in breakdown_dicts]

class BreakdownProcessor:
    def __init__(self, args, llm_model):
        self.args = args
        self.llm_model = llm_model
        self.breakdown_prompt = self._get_breakdown_prompt()

    def _get_breakdown_prompt(self):
        return "Please deconstruct the following paragraph into the smallest possible standalone self-contained facts without semantic repetition, and return the output as a jsonl, where each line is {{claim:[CLAIM], gpt-confidence:[CONF]}}.\nThe confidence score [CONF] should represent your confidence in the claim, where a 1 is obvious facts and results like 'The earth is round' and '1+1=2'. A 0 is for claims that are very obscure or difficult for anyone to know, like the birthdays of non-notable people. The input is:\n'{text_generation}'"
        
    def break_down_single(self, data, gen_id, cached_results):
        breakdown_dicts_list = []
        generation_list = [data['most_likely_generation']] + data['more_generations'][:self.args.num_samples_for_claims]
        bd_raw = []
        
        for bd_id, generation in enumerate(generation_list):
            if generation:  # no empty generation accepted
                breakdown_prompt = self.breakdown_prompt.format(text_generation=generation)
                if gen_id < len(cached_results['breakdown']) and bd_id < len(cached_results['breakdown'][gen_id]):
                    breakdown_raw_result = cached_results['breakdown'][gen_id][bd_id]
                    if generation not in breakdown_prompt:
                        with open(f'{self.args.folder_name}/breakdown_mismatch.txt', 'a') as f:
                            f.write(f'Prompt mismatch for {gen_id}th generation, {bd_id}th breakdown, {breakdown_raw_result["prompt"]} \n\n {breakdown_prompt}\n\n')
                else:
                    breakdown_raw_result = self.llm_model.generate_given_prompt(breakdown_prompt)
                    breakdown_raw_result = {'return': breakdown_raw_result, 'prompt': breakdown_prompt}
                print("Break down raw result:", breakdown_raw_result)
                breakdown_dicts = self.get_subclaims(breakdown_raw_result['return'])
                breakdown_dicts = self._clean_breakdown_dicts(breakdown_dicts)
                breakdown_dicts_list.append(breakdown_dicts)
                bd_raw.append(breakdown_raw_result)
            else:
                print("[WARNING] Empty generation")
                breakdown_dicts_list.append([])
                bd_raw.append({'return': '', 'prompt': ''})

        self._update_cached_breakdown_results(gen_id, bd_raw, cached_results)
        return breakdown_dicts_list

    def _clean_breakdown_dicts(self, breakdown_dicts):
        cleaned_dicts = []
        for dict_ in breakdown_dicts:
            if isinstance(dict_['claim'], list):
                dict_['claim'] = dict_['claim'][0]
            cleaned_dicts.append(dict_)
        return cleaned_dicts

    def _update_cached_breakdown_results(self, gen_id, bd_raw, cached_results):
        if gen_id >= len(cached_results['breakdown']):
            cached_results['breakdown'].append(bd_raw)
        elif len(bd_raw) > len(cached_results['breakdown'][gen_id]):
            cached_results['breakdown'][gen_id] = bd_raw

    def get_subclaims(self, completion):
        output = self._extract_output_from_completion(completion)

        # First attempt: try parsing as-is (maybe it's a valid JSON array or object)
        try:
            parsed = json.loads(output)
            # If it's a single object, wrap it in a list so we return a consistent array
            return parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            pass  # Will try the "wrap in array" logic below

        # Second attempt: wrap in array if it looks like multiple JSON objects
        # e.g., {...},\n{...},\n{...}
        # 1) Strip trailing commas/newlines
        trimmed_output = output.strip().rstrip(',')
        # 2) Wrap in brackets
        wrapped_output = f"[{trimmed_output}]"
        try:
            return json.loads(wrapped_output)
        except json.JSONDecodeError:
            # If still not parseable, fallback
            print("Encountered JSON decode error; falling back to parse line-by-line.")
            return self._parse_json_lines(output)

    def _extract_output_from_completion(self, completion):
        """Extract raw text output from the completion response."""
        if 'generation' in completion:
            output = completion['generation']
        else:
            output = completion['choices'][0]["message"]["content"]

        # Remove code fences
        output = (output
                .replace("```jsonl\n", "")
                .replace("```json\n", "")
                .replace("```", ""))

        # Replace keys if they’re missing quotes
        output = (output
                .replace('claim:', '"claim":')
                .replace('gpt-confidence:', '"gpt-confidence":'))

        # Attempt to isolate just the JSON part between the first '{' and the last '}'
        start = output.find('{')
        end = output.rfind('}')
        if start != -1 and end != -1:
            output = output[start:end + 1]

        return output


    def _parse_json_lines(self, output):
        """Fallback function: parse line-by-line if everything else fails."""
        results = []
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Skipping line due to JSON error: {line}")
        return results

class MatchProcessor:
    def __init__(self, args, llm_model):
        self.args = args
        self.llm_model = llm_model
        self.match_prompt = self._get_match_prompt()

    def _get_match_prompt(self):
        return """Given two lists titled "Original Claim List" and "New Claim List", your task is to integrate information from the "New Claim List" into the "Original Claim List". Please follow these detailed steps to ensure accuracy and clarity in the process:\n\nTask 1. **Verification Process:**  Your goal is to go through each statement in the "New Claim List" one by one, and determine if it is fully entailed or mentioned by any statement in the "Original Claim List." \n\nTask 2. **Compilation of Non-Entailed Claims:** Generate a list of statements from the "New Claim List" that are not already covered or implied by the "Original Claim List." For each new or unique claim that does not have an equivalent in the original list, format your output by starting each line with a dash ('-').\n\n**Original Claim List:**\n{original_claim_list}\n\n**New Claim List:**\n{new_claim_list}\n\nBegin with the Verification Process to assess each claim's relevance and uniqueness, followed by the Compilation of Non-Entailed Claims to clearly list any new insights that the "New Claim List" provides."""

    def match_single(self, gen_id, breakdown_dicts_list, cached_results):
        breakdown_dicts_list = breakdown_dicts_list.copy()
        raw_match_results = []
        total_dicts_list = breakdown_dicts_list[0]

        for j, dicts_list in enumerate(breakdown_dicts_list[1:]):
            if not dicts_list:
                # self._log_empty_breakdown(data, j)
                print(f'Empty breakdown for {gen_id}th generation, {j}th breakdown')
                continue
            
            lst_tb_merged = convert_to_claim_list(dicts_list)
            point_list = convert_to_claim_list(total_dicts_list)
            
            print(gen_id, j, len(cached_results['match']))
            match_prompt = self.match_prompt.format(
                    new_claim_list=self.invert_fact_list(lst_tb_merged),
                    original_claim_list=self.invert_fact_list(point_list)
                )
            if gen_id < len(cached_results['match']) and j < len(cached_results['match'][gen_id]):
                match_raw_result = cached_results['match'][gen_id][j]
                if match_raw_result['prompt'] != match_prompt:
                    with open(f'{self.args.folder_name}/match_mismatch.txt', 'a') as f:
                        f.write(f'Prompt mismatch for {gen_id}th generation, {j}th match\n\n{match_raw_result["prompt"]}\n\n{match_prompt}\n\n')
            else:
                match_raw_result = self.llm_model.generate_given_prompt(match_prompt)
                match_raw_result = {'return': match_raw_result, 'prompt': match_prompt}
            
            # The 'return' here is to match legacy cached results
            if 'generation' in match_raw_result['return']:
                match_result = match_raw_result['return']['generation']
            else:
                match_result = match_raw_result['return']['choices'][0]["message"]["content"]
            
            additional_point_list = self.robust_match_gpt_confidence(match_result, dicts_list)
            total_dicts_list += additional_point_list
            raw_match_results.append(match_raw_result)

        self._update_cached_match_results(gen_id, raw_match_results, cached_results)
        return total_dicts_list
    
    @staticmethod
    def invert_fact_list(fact_list):
        # Combine the facts into a string with enumeration
        enumerated_string = '\n'.join(f"{index + 1}. {fact}" for index, fact in enumerate(fact_list))
        return enumerated_string

    def robust_match_gpt_confidence(self, text, dicts_list):
        # It is not guaranteed that the text will be exactly in the same, so we need to find the closest match to record the in-line confidence
        def calculate_overlap(claim1, claim2):
            words1 = set(claim1.lower().split())
            words2 = set(claim2.lower().split())
            return len(words1.intersection(words2))

        lines = text.split('\n')
        compiled_list = []
        start_collecting = False

        for line in reversed(lines):
            if line.strip().startswith('-'):
                start_collecting = True
                item = line.strip()[1:].strip()
                item = item.split(":")[1].strip() if ":" in item else item

                highest_overlap, highest_vc = 0, -1
                for dict_ in dicts_list:
                    overlap = calculate_overlap(dict_['claim'], item)
                    if overlap > highest_overlap:
                        highest_overlap, highest_vc = overlap, dict_['gpt-confidence']
                if item:
                    compiled_list.insert(0, {'claim': item, 'gpt-confidence': highest_vc})
            else:
                if start_collecting:
                    break

        return compiled_list

    def _log_empty_breakdown(self, data, j):
        with open(f'{self.args.folder_name}/empty_breakdown.txt', 'a') as f:
            f.write(f'{data["entity"]}, {j}th Generation\n\n')

    def _update_cached_match_results(self, gen_id, raw_match_results, cached_results):
        if gen_id >= len(cached_results['match']):
            cached_results['match'].append(raw_match_results)
        elif len(raw_match_results) > len(cached_results['match'][gen_id]):
            cached_results['match'][gen_id] = raw_match_results

class AutoEvalWrapper:
    def __init__(self, folder_name, args, gpt_annotate):
        self.folder_name = folder_name
        self.args = args
        self.args.folder_name = folder_name
        self.gpt_annotate = gpt_annotate
        
        self.fact_scorer = FactScorer(data_dir=os.environ["HF_DATASETS_CACHE"], cache_dir=os.environ["HF_DATASETS_CACHE"])
        self.general_wiki_eval = General_Wiki_Eval(error_file=f'{self.folder_name}/annotate_mismatch.txt')
        self.fs_cache = []
        self.fs_raw_path = f'{self.folder_name}/annotate_raw_all.json'
        if Path(self.fs_raw_path).exists():
            with open(self.fs_raw_path, 'r') as f:
                self.fs_cache = json.load(f)
            print('Loaded from cache', self.fs_raw_path)

    def annotate_with_fact_scores(self, gen_id, data, all_claims_lst):
        if not os.path.exists(f'{self.folder_name}/annotate_mismatch.txt'):
            with open(f'{self.folder_name}/annotate_mismatch.txt', 'w') as f:
                f.write('Checking for prompt mismatches\n\n')
                
        if gen_id < len(self.fs_cache) and len(self.fs_cache[gen_id][0]) >= len(all_claims_lst):
            annotates, fs_raw = self.fs_cache[gen_id]
            
            if 'factscore' in self.args.dataset:
                for claim, fs in zip(all_claims_lst, fs_raw):
                    if claim not in fs['prompt']:
                        with open(f'{self.folder_name}/annotate_mismatch.txt', 'a') as f:
                            f.write(f'Prompt mismatch for {gen_id}th generation, {claim}\n\n{fs["prompt"]}\n\n')
        else:
            if gen_id == len(self.fs_cache) - 1 and len(self.fs_cache[gen_id][0]) < len(all_claims_lst):
                self.fs_cache = self.fs_cache[:gen_id]
                
            lst_tb_eval = all_claims_lst
            if 'factscore' in self.args.dataset:
                annotates, fs_raw = self.fact_scorer.fact_check_with_gpt(
                    topics=[data['entity']], atomic_facts=[lst_tb_eval], dataset=self.args.dataset
                )
            elif self.args.dataset in ['pop_qa_filtered', 'nq']:
                annotates, fs_raw = self.general_wiki_eval.general_wiki_eval(
                    topic=data['wiki_title'], atomic_facts=lst_tb_eval, batch_size=10
                )
            else:
                raise ValueError(f'Unknown dataset: {self.args.dataset}')
            self.fs_cache.append((annotates, fs_raw))
            with open(self.fs_raw_path, 'w') as f:
                json.dump(self.fs_cache, f, indent=2)
        return annotates

class CacheManager:
    def __init__(self, folder_name, args):
        self.raw_results_path = f'{folder_name}/match_raw_return.json'
        self.collected_results_path = f'{folder_name}/{args.dataset}_{args.model}_bipartite.json'
        self.cached_results = {'breakdown': [], 'match': []}
        self.vc_raw = []
        self.vc_raw_file = f'{folder_name}/vc_raw.json'
        self._load_cache()

    def _load_cache(self):
        if Path(self.raw_results_path).exists():
            with open(self.raw_results_path, 'r') as f:
                self.cached_results = json.load(f)
            print('Loaded from cache', self.raw_results_path)
        if Path(self.vc_raw_file).exists():
            with open(self.vc_raw_file, 'r') as f:
                self.vc_raw = json.load(f)
            print('Loaded vc raw from cache', self.vc_raw_file)

    def save_cache(self):
        with open(self.raw_results_path, 'w') as f:
            json.dump(self.cached_results, f, indent=2)
        with open(self.vc_raw_file, 'w') as f:
            json.dump(self.vc_raw, f, indent=2)

    def save_collected_results(self, breakdown_match):
        with open(self.collected_results_path, 'w') as outfile:
            json.dump(breakdown_match, outfile, indent=4)


class Break_And_Merge:
    def __init__(self, args, generations, llm_model, folder_name='', gpt_annotate=False, annotate_all=False):
        self.args = args
        self.generations = generations
        self.folder_name = folder_name
        self.llm_model = llm_model
    
        self.gpt_annotate = gpt_annotate
        self.annotate_all = annotate_all
        self.breakdown_processor = BreakdownProcessor(args, llm_model=llm_model)
        self.match_processor = MatchProcessor(args, llm_model=llm_model)
        self.fact_scorer_wrapper = AutoEvalWrapper(folder_name, args, gpt_annotate)
        self.cache_manager = CacheManager(folder_name, args)
        
        mismatch_check = ['breakdown', 'match', 'annotate']
        for check in mismatch_check:
            if not os.path.exists(f'{self.folder_name}/{check}_mismatch.txt'):
                with open(f'{self.folder_name}/{check}_mismatch.txt', 'w') as f:
                    f.write('Checking for prompt mismatches\n\n')

    def break_down_match(self):
        breakdown_match = self.generations.copy()
        for gen_id, data in tqdm(enumerate(self.generations), 'Processing Breakdown and Match'):
            # Get all the breakdown lists for the generations
            breakdown_dicts_list = self.breakdown_processor.break_down_single(data, gen_id, self.cache_manager.cached_results)
            ml_breakdown = convert_to_claim_list(breakdown_dicts_list[0])
            # Read from pre-generated claims 
            
            print(f'Processing {gen_id}th generation')
            
            # Match the breakdowns into a union list of all the claims
            all_claims_dicts = self.match_processor.match_single(gen_id, breakdown_dicts_list, self.cache_manager.cached_results)
            all_claims_lst = convert_to_claim_list(all_claims_dicts)

            if self.gpt_annotate:
                annotates = self.fact_scorer_wrapper.annotate_with_fact_scores(gen_id, data, all_claims_lst)

            self.cache_manager.save_cache()
            
            breakdown_match[gen_id]['most_likely_breakdown'] = ml_breakdown
            breakdown_match[gen_id]['most_likely_breakdown_len'] = len(ml_breakdown)
            breakdown_match[gen_id]['breakdown'] = all_claims_lst
            breakdown_match[gen_id]['breakdown_len'] = len(all_claims_lst)

            dict_list, vc_raw_list = [], []
            for claim_id in range(len(all_claims_dicts)):
                vc, vc_raw_result = self._get_verbalized_confidence(gen_id, claim_id, data, all_claims_lst[claim_id])
                vc_raw_list.append(vc_raw_result)
                
                gpt_annotation = self._get_gpt_annotation(claim_id, all_claims_lst, annotates) if self.gpt_annotate else ''
                dict_ = {
                            'claim': all_claims_dicts[claim_id]['claim'],
                            'correctness': '',
                            'inline_verbalized_confidence': all_claims_dicts[claim_id]['gpt-confidence'],
                            'gpt-score': 'S' if isinstance(gpt_annotation, dict) and gpt_annotation['objectivity'].strip().lower() in ['subjective', 's'] else gpt_annotation,
                            'gpt_annotation_result': gpt_annotation,
                            'verbalized_confidence_with_options': vc,
                        }
                dict_list.append(dict_)

            self._update_vc_raw_cache(gen_id, vc_raw_list)

            breakdown_match[gen_id]['pointwise_dict'] = dict_list
            
        self.cache_manager.save_collected_results(breakdown_match)
        return breakdown_match

    def _get_verbalized_confidence(self, gen_id, claim_id, data, new_breakdown):
        vc_raw_result = None
        if gen_id < len(self.cache_manager.vc_raw) and claim_id < len(self.cache_manager.vc_raw[gen_id]):
            vc_raw_result = self.cache_manager.vc_raw[gen_id][claim_id]

        vc, vc_raw_result = utils.get_verbalized_confidence(
            question=data['entity'],
            generation=new_breakdown,
            args=self.args,
            model_instance=self.llm_model,
            raw_result=vc_raw_result,
            problem_type='fact',
            with_options=True,
        )
        return vc, vc_raw_result

    def _update_vc_raw_cache(self, gen_id, vc_raw_list):
        if gen_id >= len(self.cache_manager.vc_raw):
            self.cache_manager.vc_raw.append(vc_raw_list)
        elif len(vc_raw_list) > len(self.cache_manager.vc_raw[gen_id]):
            self.cache_manager.vc_raw[gen_id] = vc_raw_list
        with open(self.cache_manager.vc_raw_file, 'w') as f:
            json.dump(self.cache_manager.vc_raw, f, indent=2)

    def _get_gpt_annotation(self, j, all_claims_lst, annotates):
        assert len(annotates) >= len(all_claims_lst)
        gpt_annotation_dict = annotates[j] if (self.gpt_annotate and j < len(all_claims_lst)) else ''
        if isinstance(gpt_annotation_dict, dict):
            return gpt_annotation_dict['gpt-score'] if utils.remove_non_alphanumeric(gpt_annotation_dict['content']) == utils.remove_non_alphanumeric(all_claims_lst[j]) else ''
        else:
            return gpt_annotation_dict

    def plot_auroc(self, results):
        first_n, sample_num = len(results), self.args.sc_samples
        keys_to_names = {
            f'sc_score_{sample_num}samples': 'SC',
            'verbalized_confidence_with_options': 'PH-VC',
            f'breakdown_closeness_centrality_{sample_num}samples': r'$C_C$',
            f'sc_+_vc': 'SC+VC',
        }

        dict_collection = self.collect_data_for_plotting(results=results)
        result_dict = {}
        plt.figure(0).clf()

        data_size = len(dict_collection['corr'])
        result_dict['data_size'] = data_size
        dict_bootstrap_auroc, dict_bootstrap_auprc, dict_bootstrap_auprc_n = {}, {}, {}

        for key, value in dict_collection.items():
            if key != 'corr':
                auroc, auprc, auprc_n = metrics.roc_auc_score(dict_collection['corr'], value), metrics.average_precision_score(dict_collection['corr'], value), metrics.average_precision_score(1 - dict_collection['corr'], -value)
                results = {'auroc': auroc, 'auprc': auprc, 'auprc_n': auprc_n}
                if key in keys_to_names:
                    fpr, tpr, _ = metrics.roc_curve(dict_collection['corr'], value)
                    plt.plot(fpr, tpr, label=f"{keys_to_names[key]}, auc={round(auroc, 3)}, auprc={round(auprc, 3)}, auprc_n={round(auprc_n, 3)}")
                result_dict[key] = results

        df = pd.DataFrame(dict_collection)
        plt.legend(loc=0)
        save_path = f'{self.folder_name}/plot_roc_curve_{self.args.model}_{self.args.num_samples_for_claims}matches_{sample_num + 1}samples.png'
        plt.savefig(save_path)
        save_stats_path = f'{self.folder_name}/plot_roc_curve_{self.args.model}_{self.args.num_samples_for_claims}matches_{sample_num + 1}samples.json'
        with open(save_stats_path, 'w') as f:
            json.dump(result_dict, f, indent=2)

    def collect_data_for_plotting(self, results):
        df = pd.DataFrame(results)
        sample_num = self.args.sc_samples

        corr = utils.collect_all_values(df, 'gpt-score')
        index = np.where((corr == 'Y') | (corr == 'N'))[0].astype(int)
        corr = (corr == 'Y').astype(float)
        dict_collection = {'corr': corr}
        collect_keys = [key for key in df['pointwise_dict'][0][0].keys() if key not in ["claim", "correctness", "gpt-score", "gpt_annotation_result"]]

        for key in collect_keys:
            dict_collection[key] = utils.collect_all_values(df, key)
            index = np.intersect1d(index, np.where((dict_collection[key] != -1) & (dict_collection[key] != np.inf) & (~np.isnan(dict_collection[key])))[0].astype(int))
            
        for key in dict_collection:
            dict_collection[key] = np.array(dict_collection[key])[index]

        return dict_collection
