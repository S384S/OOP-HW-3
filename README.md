# OOP-HW-3

# CSCI 3329 — Homework 3 Report
## 1. Dataset
- Dry Bean / source / 13,611 / 7
- Tabular

## 2. Preprocessing
- No Missing Values
- I believe that Gaussian NB will perform the best.
  It doesnt take a long time like neural networks or optimize
  complex functions. It simply stores averages, variances and class
  frequencies which makes training and predictions faster as well as
  minimize data usage.

## 3. Part 2 — Algorithm Comparison
| Algorithm           | Mean Accuracy   | Std  |
|Linear Classifier    |0.8994           |0.0131|
|Logistic Regression  |0.9243           |0.0062|
|KNN                  |0.9235           |0.0070|
|Gaussian NB          |0.8971           |0.0079|
|Neural Network       |0.9325           |0.0058|

## 4. Part 3 — Feature Selection
- I selected the Exhaustive Search to maximize possible accuracy.
| Algorithm           | Mean Accuracy    |
|Linear Classifier    |0.9183509205724931|
|Logistic Regression  |0.9251472000668143|
|KNN                  |0.9249632500728671|
|Gaussian NB          |0.8969510952849158|
|Neural Network       |0.9346067683980575|

## 5. Discussion
- While my hypothesis was somewhat correct, Gaussian NB definetly did
  perform the fastes. It did have a mean accuracy of 0.8971 with a
  standard deviation of 0.0079 which was slightly lower than linear classifier
  but significantly lower than the other models. However the small std means that
  Gaussian produced consistent results despite its accuracy being lower than the other
  models.
- The linear classifier achieved a 89.94% accuracy, while it is reasonobly strong, it is
  still lower than the other top performing models. It also had a std of 0.0131 which was
  higher than the others suggesting that it is more sensitive to how the data was split.
- Logistic regression performed the second best with a 92.43% accuracy. It also boasts a low
  std of 0.0062, indicating that it has good predictive capabilities, and generalization.
- KNN got third place with an accuracy of 92.53% showing that similarity patterns are very
  crucial in this dataset. Despite the higher std when compared to logistic regression at 0.0070
  shows that KNN was ever so slightly less stable.
- Gaussian NB did the worst with an accuracy of 89.71% making it the one that performed the weakest
  of the five algorithms. However it had a decent std at 0.0079 showing consisten behavior.
- Neural Network had the best accuracy at 93.25% and also the lowest std at 0.0058. Meaning it has
  great predictive capabilities and had the best performance. It's one downside is the length of time
  it takes to deliver such results.
- If the time constraint of the neural network wasn't so high, being able to have high accuracy and
  be incredibly stable would make it very worth it. But in order to have such results is the price to pay.




[Homework_3.ipynb](https://github.com/user-attachments/files/27495837/Homework_3.ipynb)

{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "source": [
        "!pip install scipy"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "hla631h_d7RM",
        "outputId": "12005b2b-f206-4a27-e292-9c10f4e6020e"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: scipy in /usr/local/lib/python3.12/dist-packages (1.16.3)\n",
            "Requirement already satisfied: numpy<2.6,>=1.25.2 in /usr/local/lib/python3.12/dist-packages (from scipy) (2.0.2)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "from scipy.io import arff"
      ],
      "metadata": {
        "id": "8RNg2BCFd_Sa"
      },
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import files\n",
        "uploaded = files.upload()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 74
        },
        "id": "_ZcBFsIWeBqW",
        "outputId": "1babc5dd-bd34-42c6-f01a-dd6b52a49e96"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              "\n",
              "     <input type=\"file\" id=\"files-77335dc8-5858-48e8-b263-fbe2b3b14033\" name=\"files[]\" multiple disabled\n",
              "        style=\"border:none\" />\n",
              "     <output id=\"result-77335dc8-5858-48e8-b263-fbe2b3b14033\">\n",
              "      Upload widget is only available when the cell has been executed in the\n",
              "      current browser session. Please rerun this cell to enable.\n",
              "      </output>\n",
              "      <script>// Copyright 2017 Google LLC\n",
              "//\n",
              "// Licensed under the Apache License, Version 2.0 (the \"License\");\n",
              "// you may not use this file except in compliance with the License.\n",
              "// You may obtain a copy of the License at\n",
              "//\n",
              "//      http://www.apache.org/licenses/LICENSE-2.0\n",
              "//\n",
              "// Unless required by applicable law or agreed to in writing, software\n",
              "// distributed under the License is distributed on an \"AS IS\" BASIS,\n",
              "// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n",
              "// See the License for the specific language governing permissions and\n",
              "// limitations under the License.\n",
              "\n",
              "/**\n",
              " * @fileoverview Helpers for google.colab Python module.\n",
              " */\n",
              "(function(scope) {\n",
              "function span(text, styleAttributes = {}) {\n",
              "  const element = document.createElement('span');\n",
              "  element.textContent = text;\n",
              "  for (const key of Object.keys(styleAttributes)) {\n",
              "    element.style[key] = styleAttributes[key];\n",
              "  }\n",
              "  return element;\n",
              "}\n",
              "\n",
              "// Max number of bytes which will be uploaded at a time.\n",
              "const MAX_PAYLOAD_SIZE = 100 * 1024;\n",
              "\n",
              "function _uploadFiles(inputId, outputId) {\n",
              "  const steps = uploadFilesStep(inputId, outputId);\n",
              "  const outputElement = document.getElementById(outputId);\n",
              "  // Cache steps on the outputElement to make it available for the next call\n",
              "  // to uploadFilesContinue from Python.\n",
              "  outputElement.steps = steps;\n",
              "\n",
              "  return _uploadFilesContinue(outputId);\n",
              "}\n",
              "\n",
              "// This is roughly an async generator (not supported in the browser yet),\n",
              "// where there are multiple asynchronous steps and the Python side is going\n",
              "// to poll for completion of each step.\n",
              "// This uses a Promise to block the python side on completion of each step,\n",
              "// then passes the result of the previous step as the input to the next step.\n",
              "function _uploadFilesContinue(outputId) {\n",
              "  const outputElement = document.getElementById(outputId);\n",
              "  const steps = outputElement.steps;\n",
              "\n",
              "  const next = steps.next(outputElement.lastPromiseValue);\n",
              "  return Promise.resolve(next.value.promise).then((value) => {\n",
              "    // Cache the last promise value to make it available to the next\n",
              "    // step of the generator.\n",
              "    outputElement.lastPromiseValue = value;\n",
              "    return next.value.response;\n",
              "  });\n",
              "}\n",
              "\n",
              "/**\n",
              " * Generator function which is called between each async step of the upload\n",
              " * process.\n",
              " * @param {string} inputId Element ID of the input file picker element.\n",
              " * @param {string} outputId Element ID of the output display.\n",
              " * @return {!Iterable<!Object>} Iterable of next steps.\n",
              " */\n",
              "function* uploadFilesStep(inputId, outputId) {\n",
              "  const inputElement = document.getElementById(inputId);\n",
              "  inputElement.disabled = false;\n",
              "\n",
              "  const outputElement = document.getElementById(outputId);\n",
              "  outputElement.innerHTML = '';\n",
              "\n",
              "  const pickedPromise = new Promise((resolve) => {\n",
              "    inputElement.addEventListener('change', (e) => {\n",
              "      resolve(e.target.files);\n",
              "    });\n",
              "  });\n",
              "\n",
              "  const cancel = document.createElement('button');\n",
              "  inputElement.parentElement.appendChild(cancel);\n",
              "  cancel.textContent = 'Cancel upload';\n",
              "  const cancelPromise = new Promise((resolve) => {\n",
              "    cancel.onclick = () => {\n",
              "      resolve(null);\n",
              "    };\n",
              "  });\n",
              "\n",
              "  // Wait for the user to pick the files.\n",
              "  const files = yield {\n",
              "    promise: Promise.race([pickedPromise, cancelPromise]),\n",
              "    response: {\n",
              "      action: 'starting',\n",
              "    }\n",
              "  };\n",
              "\n",
              "  cancel.remove();\n",
              "\n",
              "  // Disable the input element since further picks are not allowed.\n",
              "  inputElement.disabled = true;\n",
              "\n",
              "  if (!files) {\n",
              "    return {\n",
              "      response: {\n",
              "        action: 'complete',\n",
              "      }\n",
              "    };\n",
              "  }\n",
              "\n",
              "  for (const file of files) {\n",
              "    const li = document.createElement('li');\n",
              "    li.append(span(file.name, {fontWeight: 'bold'}));\n",
              "    li.append(span(\n",
              "        `(${file.type || 'n/a'}) - ${file.size} bytes, ` +\n",
              "        `last modified: ${\n",
              "            file.lastModifiedDate ? file.lastModifiedDate.toLocaleDateString() :\n",
              "                                    'n/a'} - `));\n",
              "    const percent = span('0% done');\n",
              "    li.appendChild(percent);\n",
              "\n",
              "    outputElement.appendChild(li);\n",
              "\n",
              "    const fileDataPromise = new Promise((resolve) => {\n",
              "      const reader = new FileReader();\n",
              "      reader.onload = (e) => {\n",
              "        resolve(e.target.result);\n",
              "      };\n",
              "      reader.readAsArrayBuffer(file);\n",
              "    });\n",
              "    // Wait for the data to be ready.\n",
              "    let fileData = yield {\n",
              "      promise: fileDataPromise,\n",
              "      response: {\n",
              "        action: 'continue',\n",
              "      }\n",
              "    };\n",
              "\n",
              "    // Use a chunked sending to avoid message size limits. See b/62115660.\n",
              "    let position = 0;\n",
              "    do {\n",
              "      const length = Math.min(fileData.byteLength - position, MAX_PAYLOAD_SIZE);\n",
              "      const chunk = new Uint8Array(fileData, position, length);\n",
              "      position += length;\n",
              "\n",
              "      const base64 = btoa(String.fromCharCode.apply(null, chunk));\n",
              "      yield {\n",
              "        response: {\n",
              "          action: 'append',\n",
              "          file: file.name,\n",
              "          data: base64,\n",
              "        },\n",
              "      };\n",
              "\n",
              "      let percentDone = fileData.byteLength === 0 ?\n",
              "          100 :\n",
              "          Math.round((position / fileData.byteLength) * 100);\n",
              "      percent.textContent = `${percentDone}% done`;\n",
              "\n",
              "    } while (position < fileData.byteLength);\n",
              "  }\n",
              "\n",
              "  // All done.\n",
              "  yield {\n",
              "    response: {\n",
              "      action: 'complete',\n",
              "    }\n",
              "  };\n",
              "}\n",
              "\n",
              "scope.google = scope.google || {};\n",
              "scope.google.colab = scope.google.colab || {};\n",
              "scope.google.colab._files = {\n",
              "  _uploadFiles,\n",
              "  _uploadFilesContinue,\n",
              "};\n",
              "})(self);\n",
              "</script> "
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Saving Dry_Bean_Dataset.arff to Dry_Bean_Dataset.arff\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data = arff.loadarff(\"Dry_Bean_Dataset.arff\")\n",
        "df = pd.DataFrame(data[0])"
      ],
      "metadata": {
        "id": "5FX7aLQleGT4"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df = df.applymap(lambda x: x.decode() if isinstance(x, bytes) else x)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "0x70_8GxeII-",
        "outputId": "5ae327dd-5b65-4e21-ef8e-3daf130f5a11"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/tmp/ipykernel_10212/2239276551.py:1: FutureWarning: DataFrame.applymap has been deprecated. Use DataFrame.map instead.\n",
            "  df = df.applymap(lambda x: x.decode() if isinstance(x, bytes) else x)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 8,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "DfoZK_cOdQMZ",
        "outputId": "1112ed68-1fac-4cf9-a0fc-002d467dd707"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Shape: (13611, 16) Classes: {np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4), np.int64(5), np.int64(6)}\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/tmp/ipykernel_10212/190264443.py:15: FutureWarning: DataFrame.applymap has been deprecated. Use DataFrame.map instead.\n",
            "  df = df.applymap(lambda x: x.decode() if isinstance(x, bytes) else x)\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Linear Classifier    mean=0.8994 std=0.0131\n",
            "Logistic Regression  mean=0.9243 std=0.0062\n",
            "KNN                  mean=0.9235 std=0.0070\n",
            "Gaussian NB          mean=0.8971 std=0.0079\n",
            "Neural Network       mean=0.9325 std=0.0058\n"
          ]
        }
      ],
      "source": [
        "# @title\n",
        "import pandas as pd\n",
        "from scipy.io import arff\n",
        "from sklearn.preprocessing import StandardScaler, LabelEncoder\n",
        "from sklearn.model_selection import RepeatedKFold, cross_val_score\n",
        "from sklearn.linear_model import Perceptron, LogisticRegression\n",
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "from sklearn.naive_bayes import GaussianNB\n",
        "from sklearn.neural_network import MLPClassifier\n",
        "\n",
        "# 1) Load ARFF dataset\n",
        "data = arff.loadarff(\"Dry_Bean_Dataset.arff\")\n",
        "df = pd.DataFrame(data[0])\n",
        "\n",
        "# 2) Convert byte strings to normal strings\n",
        "df = df.applymap(lambda x: x.decode() if isinstance(x, bytes) else x)\n",
        "\n",
        "# 3) Drop missing values\n",
        "df = df.dropna()\n",
        "\n",
        "# 4) Separate target and features\n",
        "y = df['Class']\n",
        "X = df.drop(columns=['Class'])\n",
        "\n",
        "# 5) Encode categorical features (if any)\n",
        "for col in X.select_dtypes(include='object').columns:\n",
        "    X[col] = LabelEncoder().fit_transform(X[col])\n",
        "\n",
        "# 6) Encode target\n",
        "if y.dtype == 'object':\n",
        "    y = LabelEncoder().fit_transform(y)\n",
        "\n",
        "# 7) Scale features\n",
        "X_scaled = StandardScaler().fit_transform(X)\n",
        "\n",
        "print('Shape:', X_scaled.shape, 'Classes:', set(y))\n",
        "\n",
        "# Models (FIXED)\n",
        "models = {\n",
        "    \"Linear Classifier\": Perceptron(),\n",
        "    \"Logistic Regression\": LogisticRegression(max_iter=1000),\n",
        "    \"KNN\": KNeighborsClassifier(),\n",
        "    \"Gaussian NB\": GaussianNB(),\n",
        "    \"Neural Network\": MLPClassifier(max_iter=1000)\n",
        "}\n",
        "\n",
        "# Cross-validation\n",
        "rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)\n",
        "\n",
        "for name, model in models.items():\n",
        "    scores = cross_val_score(\n",
        "        model, X_scaled, y,\n",
        "        cv=rkf,\n",
        "        scoring='accuracy',\n",
        "        n_jobs=-1\n",
        "    )\n",
        "\n",
        "    print(f'{name:20s} mean={scores.mean():.4f} std={scores.std():.4f}')"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "\n",
        "from sklearn.model_selection import train_test_split, GridSearchCV\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.pipeline import Pipeline\n",
        "\n",
        "from sklearn.linear_model import SGDClassifier\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "from sklearn.naive_bayes import GaussianNB\n",
        "from sklearn.neural_network import MLPClassifier\n",
        "\n",
        "\n",
        "\n",
        "# TRAIN TEST SPLIT\n",
        "\n",
        "\n",
        "X_train, X_test, y_train, y_test = train_test_split(\n",
        "    X,\n",
        "    y,\n",
        "    test_size=0.2,\n",
        "    random_state=42,\n",
        "    stratify=y\n",
        ")\n",
        "\n",
        "\n",
        "\n",
        "scaler = StandardScaler()\n",
        "\n",
        "\n",
        "# 1. LINEAR CLASSIFIER\n",
        "\n",
        "\n",
        "linear_pipeline = Pipeline([\n",
        "    ('scaler', scaler),\n",
        "    ('model', SGDClassifier(random_state=42))\n",
        "])\n",
        "\n",
        "linear_params = {\n",
        "    'model__loss': ['hinge', 'log_loss'],\n",
        "    'model__alpha': [0.0001, 0.001, 0.01],\n",
        "    'model__penalty': ['l2', 'l1']\n",
        "}\n",
        "\n",
        "linear_grid = GridSearchCV(\n",
        "    linear_pipeline,\n",
        "    linear_params,\n",
        "    cv=5,\n",
        "    scoring='accuracy',\n",
        "    n_jobs=-1\n",
        ")\n",
        "\n",
        "linear_grid.fit(X_train, y_train)\n",
        "\n",
        "print(\"\\nLINEAR CLASSIFIER\")\n",
        "print(\"Best Params:\", linear_grid.best_params_)\n",
        "print(\"Best Accuracy:\", linear_grid.best_score_)\n",
        "\n",
        "\n",
        "# 2. LOGISTIC REGRESSION\n",
        "\n",
        "\n",
        "log_pipeline = Pipeline([\n",
        "    ('scaler', scaler),\n",
        "    ('model', LogisticRegression(max_iter=5000))\n",
        "])\n",
        "\n",
        "log_params = {\n",
        "    'model__C': [0.01, 0.1, 1, 10],\n",
        "    'model__solver': ['lbfgs', 'liblinear'],\n",
        "    'model__penalty': ['l2']\n",
        "}\n",
        "\n",
        "log_grid = GridSearchCV(\n",
        "    log_pipeline,\n",
        "    log_params,\n",
        "    cv=5,\n",
        "    scoring='accuracy',\n",
        "    n_jobs=-1\n",
        ")\n",
        "\n",
        "log_grid.fit(X_train, y_train)\n",
        "\n",
        "print(\"\\nLOGISTIC REGRESSION\")\n",
        "print(\"Best Params:\", log_grid.best_params_)\n",
        "print(\"Best Accuracy:\", log_grid.best_score_)\n",
        "\n",
        "\n",
        "# 3. KNN\n",
        "\n",
        "\n",
        "knn_pipeline = Pipeline([\n",
        "    ('scaler', scaler),\n",
        "    ('model', KNeighborsClassifier())\n",
        "])\n",
        "\n",
        "knn_params = {\n",
        "    'model__n_neighbors': [3, 5, 7, 9],\n",
        "    'model__weights': ['uniform', 'distance'],\n",
        "    'model__metric': ['euclidean', 'manhattan']\n",
        "}\n",
        "\n",
        "knn_grid = GridSearchCV(\n",
        "    knn_pipeline,\n",
        "    knn_params,\n",
        "    cv=5,\n",
        "    scoring='accuracy',\n",
        "    n_jobs=-1\n",
        ")\n",
        "\n",
        "knn_grid.fit(X_train, y_train)\n",
        "\n",
        "print(\"\\nKNN\")\n",
        "print(\"Best Params:\", knn_grid.best_params_)\n",
        "print(\"Best Accuracy:\", knn_grid.best_score_)\n",
        "\n",
        "\n",
        "# 4. GAUSSIAN NB\n",
        "\n",
        "\n",
        "gnb_pipeline = Pipeline([\n",
        "    ('scaler', scaler),\n",
        "    ('model', GaussianNB())\n",
        "])\n",
        "\n",
        "gnb_params = {\n",
        "    'model__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]\n",
        "}\n",
        "\n",
        "gnb_grid = GridSearchCV(\n",
        "    gnb_pipeline,\n",
        "    gnb_params,\n",
        "    cv=5,\n",
        "    scoring='accuracy',\n",
        "    n_jobs=-1\n",
        ")\n",
        "\n",
        "gnb_grid.fit(X_train, y_train)\n",
        "\n",
        "print(\"\\nGAUSSIAN NB\")\n",
        "print(\"Best Params:\", gnb_grid.best_params_)\n",
        "print(\"Best Accuracy:\", gnb_grid.best_score_)\n",
        "\n",
        "\n",
        "# 5. NEURAL NETWORK\n",
        "\n",
        "\n",
        "nn_pipeline = Pipeline([\n",
        "    ('scaler', scaler),\n",
        "    ('model', MLPClassifier(max_iter=3000, random_state=42))\n",
        "])\n",
        "\n",
        "nn_params = {\n",
        "    'model__hidden_layer_sizes': [(50,), (100,), (50,50)],\n",
        "    'model__activation': ['relu', 'tanh'],\n",
        "    'model__alpha': [0.0001, 0.001],\n",
        "    'model__learning_rate_init': [0.001, 0.01]\n",
        "}\n",
        "\n",
        "nn_grid = GridSearchCV(\n",
        "    nn_pipeline,\n",
        "    nn_params,\n",
        "    cv=5,\n",
        "    scoring='accuracy',\n",
        "    n_jobs=-1\n",
        ")\n",
        "\n",
        "nn_grid.fit(X_train, y_train)\n",
        "\n",
        "print(\"\\nNEURAL NETWORK\")\n",
        "print(\"Best Params:\", nn_grid.best_params_)\n",
        "print(\"Best Accuracy:\", nn_grid.best_score_)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Bf9qfN93CGl3",
        "outputId": "b2ce5f97-bb22-43a1-fd4f-f683df870c78"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "LINEAR CLASSIFIER\n",
            "Best Params: {'model__alpha': 0.0001, 'model__loss': 'hinge', 'model__penalty': 'l1'}\n",
            "Best Accuracy: 0.9183509205724931\n",
            "\n",
            "LOGISTIC REGRESSION\n",
            "Best Params: {'model__C': 10, 'model__penalty': 'l2', 'model__solver': 'lbfgs'}\n",
            "Best Accuracy: 0.9251472000668143\n",
            "\n",
            "KNN\n",
            "Best Params: {'model__metric': 'euclidean', 'model__n_neighbors': 7, 'model__weights': 'distance'}\n",
            "Best Accuracy: 0.9249632500728671\n",
            "\n",
            "GAUSSIAN NB\n",
            "Best Params: {'model__var_smoothing': 1e-09}\n",
            "Best Accuracy: 0.8969510952849158\n"
          ]
        }
      ]
    }
  ]
}
