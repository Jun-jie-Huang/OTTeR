import json
def write_jsonl(data, path):
    # data是一个包含多个dict的list
    with open(path, 'w') as fp:
        for inst in data:
            fp.write(json.dumps(inst) + '\n')
def read_jsonl(path):
    # data是一个包含多个dict的list
    data = []
    with open(path, 'r') as fp:
        for line in fp:
            line = line.strip()
            data.append(json.loads(line))
    return data


def stat_ans_in_retrieved_tb():
    """
        answer string in retrieved table blocks at 1	 TB:452/2214,	 Tab:93, 	 Psg:420
        answer string in retrieved table blocks at 5	 TB:840/2214,	 Tab:220, 	 Psg:769
        answer string in retrieved table blocks at 10	 TB:1045/2214,	 Tab:318, 	 Psg:953
        answer string in retrieved table blocks at 15	 TB:1191/2214,	 Tab:383, 	 Psg:1080
        answer string in retrieved table blocks at 20	 TB:1276/2214,	 Tab:432, 	 Psg:1160
        answer string in retrieved table blocks at 50	 TB:1490/2214,	 Tab:577, 	 Psg:1372
        answer string in retrieved table blocks at 100	 TB:1619/2214,	 Tab:683, 	 Psg:1496
    :return:
    """
    path = "./models/retrieval/shared_roberta_metafine_noproj_spsgall/indexed_embeddings/dev_output_k100.json"
    dev_output = read_jsonl(path)
    # with open(path, 'r') as fp:
    #     dev_output = json.load(fp)
    print(len(dev_output))

    def get_context(tbs):
        ca = []
        ct = []
        cp = []
        for tb in tbs:
            t_str = ' [SEP] '.join(tb['table'][0]) + ' ' + ' [SEP] '.join(tb['table'][1][0])
            psg = " [SEP] ".join(tb['passages'])
            ca.append(t_str + ' ' + psg)
            ct.append(t_str)
            cp.append(psg)
        return ' '.join(ca), ' '.join(ct), ' '.join(cp)
    hit_tb = {1:0, 5:0, 10:0, 15:0, 20:0, 50:0, 100:0}
    hit_t = {1:0, 5:0, 10:0, 15:0, 20:0, 50:0, 100:0}
    hit_p = {1:0, 5:0, 10:0, 15:0, 20:0, 50:0, 100:0}
    for inst in dev_output:
        answer_str = inst['answer-text']
        for k in [1, 5, 10, 15, 20, 50, 100]:
            context_tb, context_t, context_p = get_context(inst['top_100'][:k])
            if answer_str in context_tb:
                hit_tb[k] += 1
            if answer_str in context_t:
                hit_t[k] += 1
            if answer_str in context_p:
                hit_p[k] += 1
    for i, h in hit_tb.items():
        print("answer string in retrieved table blocks at {}\t TB:{}/{},\t Tab:{}, \t Psg:{}".format(i, h, len(dev_output), hit_t[i], hit_p[i]))