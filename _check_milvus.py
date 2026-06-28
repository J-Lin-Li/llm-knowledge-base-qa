import os
os.environ.pop('HTTP_PROXY', None); os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('http_proxy', None); os.environ.pop('https_proxy', None)

from collections import Counter
from pymilvus import MilvusClient

client = MilvusClient(uri='http://127.0.0.1:19530')
collection_name = 't_collection01'

collections = client.list_collections()
print('现有 collections:', collections)

if collection_name not in collections:
    print('ERROR: collection 不存在，写入可能失败了')
else:
    stats = client.get_collection_stats(collection_name=collection_name)
    print('总记录数:', stats)

    title_res = client.query(
        collection_name=collection_name,
        filter="category == 'Title'",
        output_fields=['id'],
        limit=16384
    )
    content_res = client.query(
        collection_name=collection_name,
        filter="category == 'content'",
        output_fields=['id'],
        limit=16384
    )
    narrative_res = client.query(
        collection_name=collection_name,
        filter="category == 'NarrativeText'",
        output_fields=['id'],
        limit=16384
    )
    print(f'Title 类型 chunk 数: {len(title_res)}')
    print(f'content 类型 chunk 数: {len(content_res)}')
    print(f'NarrativeText 类型 chunk 数: {len(narrative_res)}')

    all_docs = client.query(
        collection_name=collection_name,
        filter='id > 0',
        output_fields=['filename', 'category'],
        limit=16384
    )
    filename_counter = Counter(d['filename'] for d in all_docs)
    print(f'\n各文件 chunk 数 (共 {len(all_docs)} 条):')
    for fname, cnt in sorted(filename_counter.items()):
        print(f'  {fname}: {cnt} chunks')

    # 抽查一条 content chunk 看内容是否正常
    sample = client.query(
        collection_name=collection_name,
        filter="category == 'content'",
        output_fields=['text', 'filename'],
        limit=1
    )
    if sample:
        print(f'\n抽样 content chunk (前200字):')
        print(f'  文件: {sample[0]["filename"]}')
        print(f'  内容: {sample[0]["text"][:200]}')
