from pycorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('http://localhost:9000')
ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
relevant_pos = ['VBD', 'VBN', 'VB', 'VBG', 'VBZ', 'NN', 'NNS']


def read_next(file):
    gen = []
    input = None
    gt = None
    while True:
        line = file.readline().strip()
        if line == '':
            break
        if line.startswith('Input Sentence:'):
            input = line.split('Input Sentence:')[1].strip()
        elif line.startswith('Ground Truth Sentence:'):
            gt = line.split('Ground Truth Sentence:')[1].strip()
        elif line.startswith('Generated Sentence:'):
            gen_temp = line.split('Generated Sentence:')[1].strip()
            gen.append(gen_temp)

    return input, gt, gen


def get_tokens(input_text):
    tokenized_json = nlp.annotate(input_text, properties={'annotators': 'tokenize', 'outputFormat': 'json',
                                                          'ssplit.isOneSentence': True})
    tokenized_text = []
    for tok in tokenized_json['tokens']:
        tokenized_text.append(tok['word'])
    tokenized_text = ' '.join(tokenized_text)
    return tokenized_text


def get_relevant_deps_and_context(line, args):
    dep_type = args.dependency_type
    parse = nlp.annotate(line, properties={'annotators': 'tokenize,ssplit,pos,depparse', 'outputFormat': 'json',
                                           'ssplit.isOneSentence': True})
    deps = []

    tokens = parse['sentences'][0]['tokens']
    pos = [tok['pos'] for tok in tokens]
    tokens = [tok['word'] for tok in tokens]

    for dep_dict in parse['sentences'][0][dep_type]:
        if dep_dict['dep'] not in ignore_dep:
            dep_temp = {'dep': dep_dict['dep']}
            dep_temp.update({'child': dep_dict['dependentGloss'], 'child_idx': dep_dict['dependent']})
            dep_temp.update({'head': dep_dict['governorGloss'], 'head_idx': dep_dict['governor']})
            deps.append(dep_temp)
    return tokens, pos, deps


def get_sentence_level_annotations(input, gt, gen, args):
    inp_tok, inp_pos, input_dep = get_relevant_deps_and_context(input, args)
    gt_tok, gt_pos, gt_dep = get_relevant_deps_and_context(gt, args)
    gen_tok0, gen_pos0, gen_dep0 = get_relevant_deps_and_context(gen[0], args)

    gen_tokens = []
    gen_postags = []
    gen_deps = []
    for g in gen[7:]:
        gent, genp, gend = get_relevant_deps_and_context(g, args)
        gen_tokens.append(gent)
        gen_postags.append(genp)
        gen_deps.append(gend)

    input = ' '.join(inp_tok)
    examples = []
    positive_deps_text = set([])


    # add input deps to positive dependency set
    ex = {'input': input, 'deps': [], 'context': ' '.join(inp_tok), 'sentlabel': 1}
    for dep in input_dep:
        ex['deps'].append({'dep': dep['dep'], 'label': 1,
                           'head_idx': dep['head_idx'] - 1, 'child_idx': dep['child_idx'] - 1,
                           'child': dep['child'], 'head': dep['head']})
        text = dep['dep'] + dep['child'] + dep['head']
        positive_deps_text.add(text)
    examples.append(ex)

    # add ground truth dependencies and example
    ex = {'input': input, 'deps': [], 'context': ' '.join(gt_tok), 'sentlabel': 1}
    for dep in gt_dep:
        ex['deps'].append({'dep': dep['dep'], 'label': 1,
                           'head_idx': dep['head_idx'] - 1, 'child_idx': dep['child_idx'] - 1,
                           'child': dep['child'], 'head': dep['head']})
        text = dep['dep'] + dep['child'] + dep['head']
        positive_deps_text.add(text)
    examples.append(ex)

    # add best generated paraphrase and example
    deps0_text = set([])
    ex = {'input': input, 'deps': [], 'context': ' '.join(gen_tok0), 'sentlabel': -1}
    for dep in gen_dep0:
        dep_text = dep['dep'] + dep['child'] + dep['head']
        if dep_text in positive_deps_text:
            ex['deps'].append({'dep': dep['dep'], 'label': 1,
                               'head_idx': dep['head_idx'] - 1, 'child_idx': dep['child_idx'] - 1,
                               'child': dep['child'], 'head': dep['head']})
        else:
            ex['deps'].append({'dep': dep['dep'], 'label': -1,
                               'head_idx': dep['head_idx'] - 1, 'child_idx': dep['child_idx'] - 1,
                               'child': dep['child'], 'head': dep['head']})
        text = dep['dep'] + dep['child'] + dep['head']
        deps0_text.add(text)
    examples.append(ex)

    for toks, poss, deps_list in zip(gen_tokens, gen_postags, gen_deps):
        ex = {'input': input, 'deps': [], 'context': ' '.join(toks), 'sentlabel': 0}
        neg_deps = 0
        for dep in deps_list:
            dep_text = dep['dep'] + dep['child'] + dep['head']
            if dep_text not in positive_deps_text and dep_text not in deps0_text:
                ex['deps'].append({'dep': dep['dep'], 'label': 0,
                                   'head_idx': dep['head_idx'] - 1, 'child_idx': dep['child_idx'] - 1,
                                   'child': dep['child'], 'head': dep['head']})
                neg_deps += 1
            elif dep_text in positive_deps_text:
                ex['deps'].append({'dep': dep['dep'], 'label': 1,
                                   'head_idx': dep['head_idx'] - 1, 'child_idx': dep['child_idx'] - 1,
                                   'child': dep['child'], 'head': dep['head']})
            else:
                ex['deps'].append({'dep': dep['dep'], 'label': -1,
                                   'head_idx': dep['head_idx'] - 1, 'child_idx': dep['child_idx'] - 1,
                                   'child': dep['child'], 'head': dep['head']})
        if neg_deps > 0:
            examples.append(ex)

    return examples
