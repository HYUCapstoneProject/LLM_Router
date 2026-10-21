# 11 gain lock 요약

dev에서 정한 내용이다. test 결과는 읽지 않았다. 지표는 첫 시도 이득 G = A − H(C)다.

## 비용 목표 (한 번에 고르기 규칙, 첫 시도 비용)

| 원천 | dev 행 | 25% | 50% | 75% |
|---|---:|---:|---:|---:|
| routerbench | 3101 | 0.400 | 0.600 | 0.800 |
| xroutebench | 379 | 0.400 | 0.600 | 0.800 |
| llmrouterbench | 1150 | 0.400 | 0.600 | 0.800 |
| sprout | 2961 | 0.400 | 0.600 | 0.800 |

## 방법별 선택 설정과 dev G (τ = .8)

설정 선택 기준은 τ = .8 규칙의 dev G(원천 4개 평균)다. 07b(U 기준)와 다르면 표시한다.

| 방법 | 선택 설정 | 07b 설정 | 시도 | routerbench | xroutebench | llmrouterbench | sprout | 평균 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| r1_group | `default` | `default` | 1 | +0.024 | +0.015 | +0.006 | +0.044 | +0.022 |
| knn | `n_neighbors=40` | `n_neighbors=40` | 4 | +0.030 | +0.014 | +0.000 | +0.038 | +0.020 |
| r0_group | `default` | `default` | 1 | +0.020 | +0.009 | +0.005 | +0.045 | +0.020 |
| graphrouter | `impl=official,label=success,epochs=100,checkpoint=last` **(바뀜)** | `impl=official,label=success,epochs=1000,checkpoint=last` | 4 | +0.032 | +0.006 | +0.005 | +0.033 | +0.019 |
| r1_bge | `hidden=1024,epochs=5,source_weight=off` **(바뀜)** | `hidden=1024,epochs=20,source_weight=on` | 8 | +0.030 | +0.004 | +0.007 | +0.031 | +0.018 |
| svm | `default` | `default` | 1 | +0.032 | -0.004 | +0.006 | +0.036 | +0.018 |
| routellm_retrained | `checkpoint=best_dev_loss` | `checkpoint=best_dev_loss` | 1 | +0.024 | -0.001 | +0.005 | +0.042 | +0.017 |
| irt_router | `lamda=0.3` **(바뀜)** | `lamda=0.0` | 3 | +0.031 | -0.000 | +0.003 | +0.035 | +0.017 |
| knn_a | `n_neighbors=40` **(바뀜)** | `n_neighbors=50` | 4 | +0.031 | +0.007 | +0.000 | +0.022 | +0.015 |
| r1_group_a | `default` | `default` | 1 | +0.030 | +0.013 | -0.006 | +0.024 | +0.015 |
| routerdc | `config=official_routerbench,checkpoint=final` **(바뀜)** | `config=llmrouter,checkpoint=final` | 4 | +0.013 | -0.001 | +0.013 | +0.030 | +0.014 |
| mlp | `hidden_layer_sizes=100-100-100` **(바뀜)** | `hidden_layer_sizes=100-100` | 3 | +0.009 | -0.006 | +0.001 | +0.037 | +0.010 |
| routellm_released | `default` | `default` | 1 | +0.020 | -0.010 | +0.002 | +0.027 | +0.010 |
| hybrid_llm | `router_mode=deterministic` **(바뀜)** | `router_mode=probabilistic` | 3 | +0.000 | -0.001 | +0.000 | +0.028 | +0.007 |
| mf | `epochs=3` **(바뀜)** | `epochs=5` | 2 | +0.011 | -0.022 | -0.001 | +0.038 | +0.006 |
| avengers_pro | `n_clusters=64,performance_weight=0.7` **(바뀜)** | `n_clusters=25,performance_weight=0.45` | 4 | +0.003 | -0.007 | -0.001 | +0.020 | +0.004 |

## 설정별 dev G 전체

| 방법 | 설정 | 평균 dev G |
|---|---|---:|
| avengers_pro | `n_clusters=64,performance_weight=0.7` ✓ | +0.0039 |
| avengers_pro | `n_clusters=25,performance_weight=0.45` | +0.0002 |
| avengers_pro | `n_clusters=25,performance_weight=0.7` | -0.0022 |
| avengers_pro | `n_clusters=25,performance_weight=1.0` | -0.0025 |
| graphrouter | `impl=official,label=success,epochs=100,checkpoint=last` ✓ | +0.0192 |
| graphrouter | `impl=official,label=success,epochs=1000,checkpoint=last` | +0.0146 |
| graphrouter | `impl=official,label=best,epochs=1000,checkpoint=best_f1` | +0.0030 |
| graphrouter | `impl=llmrouter` | -0.0001 |
| hybrid_llm | `router_mode=deterministic` ✓ | +0.0066 |
| hybrid_llm | `router_mode=transformed` | +0.0066 |
| hybrid_llm | `router_mode=probabilistic` | +0.0063 |
| irt_router | `lamda=0.3` ✓ | +0.0172 |
| irt_router | `lamda=0.1` | +0.0161 |
| irt_router | `lamda=0.0` | +0.0139 |
| knn | `n_neighbors=40` ✓ | +0.0204 |
| knn | `n_neighbors=50` | +0.0200 |
| knn | `n_neighbors=10` | +0.0184 |
| knn | `n_neighbors=5` | +0.0168 |
| knn_a | `n_neighbors=40` ✓ | +0.0153 |
| knn_a | `n_neighbors=50` | +0.0147 |
| knn_a | `n_neighbors=10` | +0.0094 |
| knn_a | `n_neighbors=5` | +0.0065 |
| mf | `epochs=3` ✓ | +0.0062 |
| mf | `epochs=5` | +0.0029 |
| mlp | `hidden_layer_sizes=100-100-100` ✓ | +0.0102 |
| mlp | `hidden_layer_sizes=100-100` | +0.0076 |
| mlp | `hidden_layer_sizes=100` | +0.0011 |
| r0_group | `default` ✓ | +0.0198 |
| r1_bge | `hidden=1024,epochs=5,source_weight=off` ✓ | +0.0180 |
| r1_bge | `hidden=1024,epochs=20,source_weight=on` | +0.0178 |
| r1_bge | `hidden=256,epochs=5,source_weight=off` | +0.0178 |
| r1_bge | `hidden=256,epochs=20,source_weight=off` | +0.0166 |
| r1_bge | `hidden=256,epochs=20,source_weight=on` | +0.0156 |
| r1_bge | `hidden=1024,epochs=20,source_weight=off` | +0.0154 |
| r1_bge | `hidden=256,epochs=5,source_weight=on` | +0.0143 |
| r1_bge | `hidden=1024,epochs=5,source_weight=on` | +0.0126 |
| r1_group | `default` ✓ | +0.0224 |
| r1_group_a | `default` ✓ | +0.0152 |
| routellm_released | `default` ✓ | +0.0098 |
| routellm_retrained | `checkpoint=best_dev_loss` ✓ | +0.0174 |
| routerdc | `config=official_routerbench,checkpoint=final` ✓ | +0.0138 |
| routerdc | `config=llmrouter,checkpoint=final` | +0.0121 |
| routerdc | `config=official_routerbench,checkpoint=best_dev` | +0.0100 |
| routerdc | `config=llmrouter,checkpoint=best_dev` | -0.0003 |
| svm | `default` ✓ | +0.0175 |
