import json, sys
rows=[json.loads(l) for l in open("data/originals.jsonl",encoding="utf-8")]
a,b=int(sys.argv[1]),int(sys.argv[2])
for r in rows[a:b]:
    print(f"#### {r['gid']} [{r['task']}]\n{r['original']}\n")
