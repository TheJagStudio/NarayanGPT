from flask import Flask, request, jsonify, render_template
import os
import requests
import json
from scipy import spatial
from flask_cors import CORS
import random

url = "https://gptlesson1.oss-cn-beijing.aliyuncs.com/meta.json"
response = requests.get(url)
apiKeys = response.json()["APIkey"]["keys"]

with open("embeddingData.json", "r", encoding="utf-8") as f:
    df = json.load(f)


def embeddingGen(query):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer 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",  # apiKeys[len(apiKeys) - 1],
    }
    json_data = {
        "input": query,
        "model": "text-embedding-ada-002",
    }

    response = requests.post(
        "https://api.openai.com/v1/embeddings", headers=headers, json=json_data
    )
    return response.json()


def strings_ranked_by_relatedness(query, df, top_n=5):
    relatedness_fn = lambda x, y: 1 - spatial.distance.cosine(x, y)
    query_embedding_response = embeddingGen(query)
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"])) for row in df
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]


app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/api/getAPI", methods=["POST"])
def getAPI():
    return jsonify({"API": "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"})


@app.route("/api/getContext", methods=["POST"])
def getContext():
    question = request.form["question"]
    strings, relatednesses = strings_ranked_by_relatedness(question, df, top_n=2)
    context = "\n---------\n".join(strings)
    return jsonify({"context": context})


if __name__ == "__main__":
    app.run(debug=True)
