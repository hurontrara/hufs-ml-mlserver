final_data.csv : 분석할 데이터 (소분노 포함)
ko.bin : pre-trained word2vec
pca_model.pkl : 200 -> 4차원으로 축소시켜주는 모델
KMEANS_model : 21개 cluster로 학습시킨 모델
cluster_result.csv : 각 단어별 클러스터값 부여된 dataframe
sorted_result.csv : cluster_result.csv를 cluster번호에 따라 정렬한 dataframe
final_model.ipynb : 분석 코드 