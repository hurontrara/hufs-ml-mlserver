from flask import Flask, Response, request, jsonify, make_response
import joblib
import gensim
import traceback
import numpy as np

app = Flask(__name__)
kMeansModel = joblib.load('../../muchu-mlserver/API_SERVER/KMEANS-200vec-75cluster.txt')
W2Vmodel = gensim.models.Word2Vec.load('ko.bin')


# PCA
# 로컬 이슈
# EC2 관련해서 잘 돌아가도록


@app.route('/k-means', methods=['POST'])
def modeling():
    try:
        data = request.get_json()
        word = data['word']

        wordToVector = W2Vmodel[word]  # shape  (1, 200)
        wordToVector = wordToVector.reshape((1, 200))
        wordToVector = wordToVector.astype(np.double)
        outputClusterNum = kMeansModel.predict(wordToVector)[0]

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
