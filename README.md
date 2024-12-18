# Seedtag Codetest: MLOps

**Captain Andor**,

First of all, I would like to congratulate you in name of the Alliance to Restore the Republic; your work in Cato Neimoida was stellar, a great example of what we need in these perilous days. But the fight is not over yet, and we need to entrust you one more mission.

Our Bothan Spynet, the Galactic Empire has developed an interest on Kyber Crystals. As you may know, Ilum is one of the harvesting planets that the Jedi used back in Republic times to get crystals for their lightsabers. We deployed a number of modified Gonk Droids to sample and analyze crystal ores, in hopes to find the true colors of the crystal, which can be **blue**, **green** or **yellow**.

Unfortunately, these droids don't have the processing power needed to perform said analysis. This is where you come into play.

## Mission Details

Our team of Machine Learning Engineers have developed **two models** based on different approaches: One developed with [Scikit-Learn](https://scikit-learn.org/stable/) and one with [PyTorch](https://pytorch.org/).

These models will take one or multiple samples of data collected by the droids and produce an output with the probabilities for each of the colors. We need you to develop a REST API that serves these models so that the droids can send the crystal data collected by them.

Given the difficulties we had to update the droids' software, some of them can only request analysis to the Scikit-Learn model and some others can only request analysis to the PyTorch model.


We need the following endpoints in our service:

* **/sklearn**, where the Gonk droids designed for Scikit-Learn will send the data.
* **/pytorch**, where the Gonk droids designed for PyTorch will send the data.

In addition, we need an extra endpoint for our astromech droids. These astromechs will include in the request an extra parameter that will specify the model that they want to use:

* **/astromech**, where astromech droids will select themselves the model.

These droids only work sending data to the 3000 port, so make sure that **your service listens to  port 3000**

### Requests

The Gonk droids will use a **POST** request with the following body:

```json
{
  "crystalData": [[0.92, 0.12, 0.31, 0.09]]
}
```

```json
{
  "crystalData": [
    [0.92, 0.12, 0.31, 0.09],
    [0.31, 0.54, 0.78, 0.17],
    [0.38, 0.04, 0.21, 0.98]
    ]
}
```

In similar fashion, the astromechs will use a **POST** request and the following body:

```json
{
  "crystalData": [[0.92, 0.12, 0.31, 0.09]],
  "model": "pytorch"
}
```

**The model field can take two valid values: "pytorch" or "sklearn"**.

### Responses

Our models don't have any labels in the answer. These are attached in a file in this codetest, but just in case, the labels are the following:

```python
["blue", "green", "yellow"]
```

These correspond to the following outputs in the models:

```python
[0, 1, 2]
```

You should be able to include these labels in the responses without modifying the models themselves.

As a response, the REST API should answer with the following structure:

* One field called **prediction** which will output a list with the label of the most probable result in each of the input data.
* One field called **scores**, which will have a list of dictionaries with the score for each of the labels.

e.g.

* For **single predictions**:

```json
{
  "prediction": [
    "blue"
  ],
  "scores": [
    {
      "blue": 0.9799801195910143,
      "green": 0.02001979984409025,
      "yellow": 8.056489543610177e-8
    }
  ]
}
```

* For **batch predictions**

```json
{
  "prediction": [
    "blue",
    "blue",
    "green"
  ],
  "scores": [
    {
      "blue": 0.9927769331813387,
      "green": 0.007223053394387628,
      "yellow": 1.3424273700413213e-8
    },
    {
      "blue": 0.9924667517303258,
      "green": 0.007533175370560812,
      "yellow": 7.28991131694263e-8
    },
    {
      "blue": 0.007533175370560812,
      "green": 0.9954911544544908,
      "yellow": 1.0907653719137907e-7
    }
  ]
}
```

## Additional details

* Droids are prone to failure. The data may be corrupt. In case you detect any anomaly with the data (either missing or extra dimensions, other type of values, etc), you must return an HTTP Status Code [422](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/422).
* Astromechs can fail too! Make sure that if the model they choose is not "sklearn" or "pytorch", send a 400 error.
* In case the model prediction fails, return an HTTP Status Code 500.
* Rembember, this project should be easy to maintain and upgrade. We greatly value proficiency in Object Oriented Programming, testing and clean code.
* We have included a requirements.txt for you to run the models, as well as some example code to use them. You are free to update it with the libraries you need.
* To help you with the expected results, we have included a test file for the cases that we listed above. If you want to use it, just need to install Pytest and Requests first (already included in the requirements.txt)


## Code Delivery Instructions

Your soultion to this problem, along with an analysis and evalutaion, should be packed into a zip file named <name>_codetest_mlops_seedtag.


Thank your for your attention, Captain.
**May the Force be with you**
