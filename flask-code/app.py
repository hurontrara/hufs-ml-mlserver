from flask import Flask, Response, request, jsonify, make_response
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib
import gensim
import numpy as np
import pickle

app = Flask(__name__)
W2Vmodel = gensim.models.Word2Vec.load('ko.bin')
with open("KMEANS_model", 'rb') as f:
    kMeansModel = pickle.load(f)
with open('pca_model.pkl', 'rb') as pca_file:
    pca_model = pickle.load(pca_file)


@app.route('/k-means', methods=['POST'])
def modeling():
    try:
        data = request.get_json()
        word = data['word']

        wordToVector = W2Vmodel[word]  # shape  (1, 200)
        wordToVector = wordToVector.reshape((1, 200))
        input_pca = pca_model.transform(wordToVector)
        outputClusterNum = kMeansModel.predict(input_pca)

        responseDict = {'clusterNum': "{}".format(outputClusterNum)}
        response = make_response(responseDict)
        response.status_code = 200

        return response

    except KeyError:

        responseDict = {'errorMessage': "존재하지 않는 단어입니다."}
        response = make_response(responseDict)
        response.status_code = 400

        return response

    except:
        responseDict = {'errorMessage': "서버 내부 에러입니다."}
        response = make_response(responseDict)
        response.status_code = 500

        return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
