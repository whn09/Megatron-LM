import json
import os
import webdataset as wds

from tqdm import tqdm

llava_pretrain_dir = '/workspace/dataset/LLaVA-NeXT-Data/llava_next_raw_format'

# Paths to the dataset files
# json_file = os.path.join(llava_pretrain_dir, 'llava_next_raw_format_processed.json')
json_file = os.path.join(llava_pretrain_dir, 'llava_v1_5_mix665k.json')
output = os.path.join(llava_pretrain_dir, 'wds')

if not os.path.exists(output):
    os.mkdir(output)

# Load data
with open(json_file, 'r') as f:
    data = json.load(f)
    
stats = {}
ids = {}
skip_cnt = 0
with wds.ShardWriter(os.path.join(output, 'pretrain-%d.tar'), maxcount=10000) as shard_writer:
    for entry in tqdm(data):
        # if 'llava_pretrain_lcs558k' in entry['image']:
        if 'image' in entry:
            pathname = entry['image'].split('/')[0]
            if pathname not in stats:
                stats[pathname] = 0
            stats[pathname] += 1
            
            if entry['id'] not in ids:
                ids[entry['id']] = entry['image']
            else:
                # print('Skip:', entry)
                skip_cnt += 1
                continue
            
            try:
                with open(os.path.join(llava_pretrain_dir, entry['image']), "rb") as img_file:
                        image_data = img_file.read()
                sample = {
                    "__key__": entry['id'],
                    "jpg": image_data,
                    "json": json.dumps(entry['conversations']).encode("utf-8"),
                }
                shard_writer.write(sample)
            except Exception as e:
                print('Error:', entry['image'], e)

print('stats:', stats)
print('skip_cnt:', skip_cnt)
print(f"Dataset successfully converted to wds")
