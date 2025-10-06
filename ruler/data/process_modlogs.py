import json
import os
import re
from tqdm import tqdm
from wordllama import WordLlama
import pickle as pkl
from collections import defaultdict
from ruler.data.constants import URL_REPLACEMENT_TOKEN
from ruler.data.text_utils import markdown_to_text, replace_urls



def encode_modlogs(entry, rules, applied_rule_text):
    return {
        'ap_id': entry['comment']['ap_id'],
        'content': entry['comment']['content'],
        'removed': entry['comment']['removed'],
        'applied_rule_n': entry['applied_rule_n'],
        'applied_rule_text': applied_rule_text,
        'reason': entry['mod_remove_comment']['reason'],
        'community': {
            'actor_id': entry['community']['actor_id'],
            'rules': rules,
            'description': entry['community']['description'],
            'name': entry['community']['name'],
            'nsfw': entry['community']['nsfw'],
        },
        'instance': entry['instance'],
        'modlog_id': entry['mod_remove_comment']['id'],
        'mod_person_id': entry['mod_remove_comment']['mod_person_id'],
    }


def extract_rules_from_reason(reason, description, ruleset):
    if description:
        for rules_ in ruleset:
            rules = rules_['rules']       
            if description == rules_['description']:
                #try:
                    #check rule number match
                numbers = re.findall(r'\d+', reason)
                if len(numbers) ==1:
                    rulenum = numbers[0]
                    for r in rules['rules'].keys():
                        if r == str(rulenum):
                            return rules, int(rulenum)

                #check reason substring match
                matched_reason_substrings = set()
                for r in rules['rules'].keys():
                    if reason.lower() in rules['rules'][r].lower():
                        matched_reason_substrings.add(int(r))

                if len(matched_reason_substrings)==1:
                    return rules, list(matched_reason_substrings)[0]

                return rules, -1
                #except:
                #    return rules, None
        return rules, None
                #raise NotImplemented
    else:
        for rules_ in ruleset:
            rules = rules_['rules']           
            #try:
                #check rule number match
            numbers = re.findall(r'\d+', reason)
            if len(numbers) ==1:
                rulenum = numbers[0]
                for r in rules['rules'].keys():
                    if r == str(rulenum):
                        return rules, int(rulenum)

            #check reason substring match
            matched_reason_substrings = set()
            for r in rules['rules'].keys():
                if reason in rules['rules'][r]:
                    matched_reason_substrings.add(int(r))

            if len(matched_reason_substrings)==1:
                return rules, list(matched_reason_substrings)[0]

            return rules, -1
            #except:
            #    return rules, None
        return rules, None
            #raise NotImplemented



def main(data_dir='../../data', storage_dir='', min_char=30, max_char=400):

    print("opening descriptions and rules")
    modlog_folder = os.path.join(data_dir, 'raw', 'modlogs')
    community_rules = defaultdict(list)
    with open(os.path.join(data_dir, 'interim', 'v2_communities_with_all_rules.jsonl'), encoding='utf8') as f:
        for line in f:
            job = json.loads(line)
            tempd = {}
            if job['nrules'] > 1:
                tempd['description'] = job['description']
                tempd['rules'] = job['rules']
                tempd['nrules'] = job['nrules']
                community_rules[job['actor_id']].append(tempd)



    #community_rules = {community['actor_id']: community['rules'] for community in communities}
    all_communities = list(set(list(community_rules.keys())))
    print("Number of communities: ", len(all_communities))

    entries = list()
    processed_comments = set()

    ##wordllama for fuzzymatch
    #wl = WordLlama.load()
    comments = set()

    community_drop=0
    unusual_removal_drop=0
    no_reason_drop=0
    mass_removed_drop=0
    no_description_drop=0
    rulemap_fail_drop=0
    url_removal_drop=0


    print("fetching modlogs")
    with open(os.path.join(data_dir, 'interim', 'v2_binary_modlogs.jsonl'), 'w+', encoding='utf8') as fw:
        with open(os.path.join(data_dir, 'interim', 'v2_nonbinary_modlogs.jsonl'), 'w+', encoding='utf8') as fnb:
            for instance in tqdm(os.listdir(modlog_folder)):
                modlog_file = os.path.join(modlog_folder, instance, 'removed_comments.jsonl')
                if not os.path.exists(modlog_file): continue
                with open(modlog_file, encoding='utf8') as f:
                    for entry in map(json.loads, f):
                        if "comment" not in entry: continue
                        if "ap_id" not in entry['comment']: continue
                        comments.add(entry['comment']['ap_id'])
        
                        community = entry['community']['actor_id']
                        if community not in all_communities: continue
                        community_drop+=1

                        comment_id = entry['comment']['ap_id']
                        if comment_id in processed_comments: continue
                        # comments w/ remove=true and deleted=false
                        #if not entry['comment']['removed']: continue
                        if entry['comment']['deleted']: continue

                        # comments whose body is not removed (string match with *Permanently deleted*, mass removed, etc)
                        if entry['comment']['content'].strip().lower() == '*permanently deleted*': continue
                        # comments that are not mass-removed
                        unusual_removal_drop+=1
                        if "reason" not in entry['mod_remove_comment'] : continue
                        #if "mass removed"  in entry['mod_remove_comment']['reason'].strip().lower() : continue

                        if not entry['mod_remove_comment']['reason']: continue
                        no_reason_drop+=1

                        if "mass removed" in entry['mod_remove_comment']['reason'].lower(): continue
                        if "reversal of content removal" in entry['mod_remove_comment']['reason'].lower(): continue
                        if "qualitycontrol bot" in entry['mod_remove_comment']['reason'].lower(): continue

                        mass_removed_drop+=1
                        if 'description' not in entry['community']: continue

                        if not entry['community']['description']: continue
                        no_description_drop+=1
                        
                        rules, extracted_rules = extract_rules_from_reason(entry['mod_remove_comment']['reason'], None, community_rules[community])
                        #if len(extracted_rules) > 1: continue
                        if not extracted_rules: continue
                        entry['applied_rule_n'] = int(extracted_rules)#.pop()
                        # comments w/ more than just urls
                        rulemap_fail_drop+=1

                        entry['comment']['content'] = replace_urls(markdown_to_text(entry['comment']['content']),
                                                                URL_REPLACEMENT_TOKEN)
                        if len(''.join(entry['comment']['content'].split(URL_REPLACEMENT_TOKEN)).strip()) == 0: continue
                        # comments w/ 10<=chars excluding urls<=400 (decide after plotting)
                        #if not (min_char <= len(entry['comment']['content']) <= max_char): continue

                        url_removal_drop+=1

                        entry['instance'] = instance

                        if extracted_rules==-1:
                            applied_rule_text = "None"
                        else:
                            applied_rule_text = rules['rules'][str(entry['applied_rule_n'])]
                        
                        curr_entry = encode_modlogs(entry, rules, applied_rule_text)
                        #entries.append(encode_modlogs(entry, community_rules))
                        processed_comments.add(comment_id)
        
            
                        fw.write(json.dumps(curr_entry, sort_keys=True, ensure_ascii=False) + '\n')
                        if extracted_rules!=-1:
                            fnb.write(json.dumps(curr_entry, sort_keys=True, ensure_ascii=False) + '\n')

    print("Comment drops by reason")
    print("Community drop survival: ", str(community_drop))
    print("Unusual removal survival: ", str(unusual_removal_drop))
    print("Reason drop survival: ", str(no_reason_drop))
    print("Mass removal survival: ", str(mass_removed_drop))
    print("Description drop survival: ", str(no_description_drop))
    print("Rulemap drop survival: ", str(rulemap_fail_drop))
    print("URL drop survival: ", str(url_removal_drop))


    with open(os.path.join(data_dir, 'interim', 'v2_all_modlogs.pkl'), 'wb') as pfile:
        pkl.dump(comments, pfile)


if __name__ == '__main__':
    main()
