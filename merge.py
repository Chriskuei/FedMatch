import json

first = open('pred/inqa_random_pred_0v12.json')
second = open('pred/inqa_random_pred_1v2.json')

dataset1 = [json.loads(line) for line in first]
dataset2 = [json.loads(line) for line in second]

out_file = open('pred/inqa_merge_0v12_1v2.json', 'w')

for datum1, datum2 in zip(dataset1, dataset2):
    if datum1['pred'] == 1:
        datum1['pred'] = datum2['pred'] + 1
        # if datum1['pred'] == 1:
            # datum1['pred'] = 2
    # elif datum1['pred'] == 1:
    #     datum1['pred'] = 2
    out_file.write(f'{json.dumps(datum1, ensure_ascii=False)}\n')