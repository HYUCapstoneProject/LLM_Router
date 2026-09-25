"""Add independently verified audit visuals to the unchanged validation report."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def read(path):
    return json.loads(path.read_text())

def verify(root):
    manifest=read(root/'manifest.json')
    for name,record in manifest['files'].items():
        path=root/name
        if sha(path)!=record['sha256'] or path.stat().st_size!=record['size_bytes']:
            raise ValueError(f'Existing report checksum mismatch: {name}')
    return manifest

def block(page,name,content):
    wrapped=f'<!-- {name}:start -->{content}<!-- {name}:end -->'
    pattern=rf'<!-- {name}:start -->.*?<!-- {name}:end -->'
    if re.search(pattern,page,flags=re.S):
        return re.sub(pattern,lambda _:wrapped,page,flags=re.S)
    return page.replace('</main>',wrapped+'</main>')

def main(args):
    root=args.diagnostics_dir.resolve()
    old=verify(root)
    for directory in ['category-baselines','category-permutation']:
        verify(root/directory)
    source=read(root/'category-baselines/original-audit.json')
    xrb=source['by_source']['xroutebench']
    rb=source['by_source']['routerbench']
    root_index=root/'index.html'
    page=root_index.read_text()
    category=f'''<section id="category-baselines"><h2>R0: 카테고리의 다수 라벨만 알아도 설명되는가?</h2><p>학습 데이터의 원천별 다수 라벨, 원천·카테고리별 다수 라벨을 각각 고정해 개발검증에 적용했습니다. 두 기준선은 검증 정답으로 규칙을 맞추지 않았습니다. 비교 대상은 비용 정책 적용 전 R0의 난이도 분류이며 R1·R2에는 이 정확도를 적용하지 않습니다.</p><a href="category-baselines/r0-train-category-baselines.png"><img src="category-baselines/r0-train-category-baselines.png" alt="Train-only majority 기준선과 R0의 정확도, macro F1, balanced accuracy"></a><p>xRouteBench 정확도는 카테고리 기준선 {xrb['source_category_majority']['accuracy']:.2%}, R0 {xrb['r0_raw_argmax']['accuracy']:.2%}로, R0의 추가 정답은 459건 중 3건입니다. Macro F1은 기준선 {xrb['source_category_majority']['macro_f1_fixed_3_labels']:.4f}가 R0 {xrb['r0_raw_argmax']['macro_f1_fixed_3_labels']:.4f}보다 높습니다. RouterBench는 기준선 {rb['source_category_majority']['accuracy']:.2%}에서 R0 {rb['r0_raw_argmax']['accuracy']:.2%}로 개선됩니다.</p><p><b>높은 원천별 정확도만으로 질문의 난이도를 일반화했다고 결론 낼 수 없습니다.</b> 단순한 카테고리 정보가 일부 정확도를 설명하지만, R0 전체가 카테고리 규칙에만 의존한다는 인과 증명도 아닙니다. 카테고리별 표본 수와 다른 지표를 함께 봐야 합니다.</p><details><summary>카테고리별 정확도 차이</summary><a href="category-baselines/r0-category-differences.png"><img src="category-baselines/r0-category-differences.png" alt="카테고리별 R0 정확도에서 train majority 정확도를 뺀 차이"></a><p>표본이 작은 구간의 큰 차이는 과해석하지 않습니다. 0은 같은 정확도이며 반드시 동일한 예측을 뜻하지는 않습니다.</p></details><p><a href="category-baselines/original-audit.md">독립 분석 메모</a> · <a href="category-baselines/original-audit.json">전체 지표·train prior·입력 SHA</a> · <a href="category-baselines/original-audit.parquet">행별 예측</a> · <a href="category-baselines/original-audit.py">분석 재현</a> · <a href="category-baselines/render.py">시각화·독립 검증</a> · <a href="category-baselines/manifest.json">파일 manifest</a></p></section>'''
    page=block(page,'category-baseline-audit',category)
    permutation='''<section id="category-permutation"><h2>사후 진단: 카테고리 안에서 선택을 바꿔도 비슷한가?</h2><p>동일한 원천·카테고리 안에서 고정 budget 0.5의 선택을 1,000회 재배정했습니다. 각 모델의 low/medium/high 사용 건수는 매번 그대로입니다. 모든 모델에는 같은 행 permutation을 적용했습니다. 막대는 재배정 결과의 중앙 95% 범위이고 점은 실제 고정 정책입니다. 각 행의 셔플 평균을 0으로 놓았습니다.</p><a href="category-permutation/posthoc-quality-cost-ranges.png"><img src="category-permutation/posthoc-quality-cost-ranges.png" alt="카테고리 내부 선택 재배정의 품질·비용 범위와 실제 선택 비교"></a><p><b>이 범위는 신뢰구간이 아닙니다.</b> 이미 모델·정책 선택에 사용한 개발검증의 사후 진단이며, 독립 test의 유의성 검정도 아닙니다. 선택 비율을 유지해도 문항별 비용이 달라 실현 비용이 바뀝니다. 각 원천 비용 단위 그대로 표시했으며, 원천 간 비용을 평균하거나 현재 서비스 비용으로 환산하지 않았습니다.</p><p>RouterBench에서는 R1·R2의 품질이 셔플 범위 위에 있어 카테고리별 모델 사용 비율 이상의 질문별 선택 정보가 있다는 보조 근거입니다. 하지만 비용도 범위 위이므로 같은 비용에서 더 낫다는 뜻은 아닙니다. xRouteBench에서는 세 모델의 품질이 모두 범위 안에 있습니다. 이 진단만으로 출처·카테고리 규칙에 대한 의존을 배제하거나 R1의 질문별 선택 우위를 일반화할 수 없습니다.</p><p><a href="category-permutation/original-audit.md">해석 메모</a> · <a href="category-permutation/original-audit.json">1,000회 원시 집계·지표·SHA</a> · <a href="category-permutation/ranges.csv">수치 표</a> · <a href="category-permutation/original-audit.py">분석 재현</a> · <a href="category-permutation/render.py">시각화·분포 검증</a> · <a href="category-permutation/manifest.json">파일 manifest</a></p></section>'''
    page=block(page,'category-permutation-audit',permutation)
    arena_path=root/'routerarena/visuals.json'
    if arena_path.exists():
        verify(root/'routerarena')
        arena=read(arena_path)
        if arena['synthetic_only'] or not arena['prepared_metadata_verified']:
            raise ValueError('Do not integrate synthetic or unjoined RouterArena data')
        full=arena['full_independent_rows'];paired=arena['paired_variants']
        body=f'''<section id="routerarena"><h2>RouterArena: 공식 라벨과 R0·R1·R2의 선택</h2><p>정제된 full {full:,}건과 대응하는 robustness {paired:,}쌍을 고정 λ=0·최대 512토큰으로 확인했습니다. 세 모델의 같은 질문 ID와 공식 라벨을 검증했습니다. <a href="routerarena/index.html"><b>전체 카테고리·도메인·medium·robustness 보고서 열기 →</b></a></p><a href="routerarena/conditional-on-official.png"><img src="routerarena/conditional-on-official.png" alt="RouterArena 공식 난이도별 R0와 R1·R2 action 슬롯 분포"></a><p><b>R0는 난이도 분류기이고 R1·R2는 action 슬롯의 argmax 투영입니다.</b> R1·R2 표는 난이도 학습 성능, 후보별 실측 성공률, 운영 경로 또는 공식 Arena Score가 아닙니다. 개발검증의 T1 정답률 기반 라벨과 Arena 공식 Difficulty는 정의가 다르므로 정확도 차이를 같은 과제의 성능 하락이라고 단정하지 않습니다.</p><p>과거 RouterArena 평가를 학습 전에 확인한 이력이 있어 완전히 보지 않은 blind test가 아닙니다. Full과 robustness는 같은 원본의 쌍이므로 합산하지 않습니다. 이번 진단으로 모델·정책을 다시 선택하거나 배포하지 않았습니다.</p><p><a href="routerarena/visuals.json">전체 수치·검증 내역</a> · <a href="routerarena/manifest.json">산출 manifest</a></p></section>'''
        page=re.sub(r'<section><h2>RouterArena 범위</h2>.*?</section>','',page,flags=re.S)
        page=block(page,'routerarena-audit',body)
        page=page.replace('기존 R0·R1·R2 검증 점수를 재집계했으며, 모델 추론이나 정책 재선택은 수행하지 않았습니다.','개발검증은 기존 R0·R1·R2 점수를 재집계했습니다. RouterArena는 별도의 고정 설정 추론 결과로 구분했고 정책을 다시 선택하지 않았습니다.')
    links='<nav class="downloads" aria-label="보고서 바로가기"><a href="#category-baselines">R0 train-only 기준선</a><a href="#category-permutation">카테고리 내부 셔플</a>'
    if arena_path.exists():
        links+='<a href="#routerarena">RouterArena</a><a href="routerarena/index.html">RouterArena 상세 그래프</a>'
    links+='</nav>'
    page=re.sub(r'<!-- integrated-nav:start -->.*?<!-- integrated-nav:end -->','',page,flags=re.S)
    page=page.replace('</header>','</header><!-- integrated-nav:start -->'+links+'<!-- integrated-nav:end -->')
    root_index.write_text(page)
    if Path(__file__).resolve()!=root/'integrate-report.py':
        shutil.copyfile(__file__,root/'integrate-report.py')
    old['integration_script_sha256']=sha(root/'integrate-report.py')
    old['subreports']={str(p.parent.relative_to(root)):sha(p) for p in root.rglob('manifest.json') if p.parent!=root}
    old['files']={str(p.relative_to(root)):{'sha256':sha(p),'size_bytes':p.stat().st_size} for p in sorted(root.rglob('*')) if p.is_file() and p!=root/'manifest.json'}
    (root/'manifest.json').write_text(json.dumps(old,indent=2)+'\n')
    verify(root)
    print(json.dumps({'output':str(root_index),'files':len(old['files']),'routerarena_integrated':arena_path.exists()}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostics-dir',type=Path,required=True)
    main(parser.parse_args())
